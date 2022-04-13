import gzip
from collections import defaultdict

import jsonlines
import pandas as pd
import spacy
from tqdm import tqdm

from .qa_utils import parse_span
# from srl_as_qa_parser import TemplateQuestionGenerator

FAST_DEV_RUN_THRESHOLD = 100


def load_tsv_samples(tsv_path, fast_dev_run):
    """
        Reads a TSV file with the following columns: question   gold_answer_span    text
        :param tsv_path:
        :param fast_dev_run:
        :return:
        """
    sep = "\t"
    if tsv_path.endswith(".csv"):
        sep = ","
    nrows = FAST_DEV_RUN_THRESHOLD if fast_dev_run else None
    # Watch out for a subtle issue with NA values.
    # Strings such as NA are used by pandas to signal a missing value in the dataframe
    # But in our case, we shouldn't have missing values at all,
    # and NA can be a name of a protein
    df = pd.read_csv(tsv_path, sep=sep, nrows=nrows, keep_default_na=False)
    records = df.to_dict(orient="records")
    has_gold_span = 'gold_answer_spans' in records[0]
    has_gold_answer = 'gold_answers' in records[0]
    if has_gold_span:
        for r in records:
            spans = r['gold_answer_spans'].split("~!~")
            r['gold_answer_spans'] = [parse_span(sp) for sp in spans]
    if has_gold_answer:
        for r in records:
            r['gold_answers'] = r['gold_answers'].split("~!~")

    return records


def yield_mrqa_samples(record):
    if 'header' in record:
        return
    tokens = [t[0] for t in record['context_tokens']]
    for qa in record['qas']:
        # For each question, one or more possible answers
        # For training, take the first answer.
        # For evaluation, take all answer instances.
        question = qa['question']
        gold_answer_spans = []
        for answer in qa['detected_answers']:
            token_spans = sorted(answer['token_spans'])
            for token_start, token_end in token_spans:
                # make ends exclusive
                token_end += 1
                gold_answer_spans.append((token_start, token_end))
        gold_answer_spans = sorted(set(gold_answer_spans))
        gold_answers = [" ".join(tokens[start: end]) for start, end in gold_answer_spans]
        # take this for training.
        token_start, token_end = gold_answer_spans[0]
        yield {'text': " ".join(tokens),
               'question': question,
               'answer_start': token_start,
               'answer_end': token_end,
               'gold_answers': gold_answers,
               'gold_answer_spans': gold_answer_spans,
               'doc_id': qa["qid"]}


def load_mrqa_samples(mrqa_path, fast_dev_run=False):
    samples = []
    with gzip.open(mrqa_path) as gzip_input:
        with jsonlines.Reader(gzip_input) as json_reader:
            for idx, record in tqdm(enumerate(json_reader), desc="Reading MRQA Dataset"):
                if fast_dev_run and (idx >= FAST_DEV_RUN_THRESHOLD):
                    break
                for s in yield_mrqa_samples(record):
                    samples.append(s)
    return samples


def verify_sent_id(sample):
    if 'sent_idx' in sample:
        sample['sent_id'] = sample['sent_idx']
        del sample['sent_idx']
    return sample


# class SrlRoleLinkingLoader:
#     def __init__(self, generator: TemplateQuestionGenerator):
#         # spacy is a great lemmatizer.
#         self.nlp = spacy.load('en_core_web_sm',
#                               disable=['parser', 'ner', 'tagger'])
#         self.generator = generator
#
#     def get_lemma(self, word):
#         return self.nlp([word])[0].lemma_
#
#     def yield_qa_inputs(self, frame):
#         predicate = frame["predicate"]
#         pred_text = predicate["text"].lower()
#         pred_span = predicate['span']
#         doc_id = frame['doc_id']
#         sent_id = frame['sent_id']
#         predicate_lemma = self.get_lemma(pred_text)
#         role_to_args = defaultdict(list)
#         for arg in frame['arguments']:
#             role = arg['role_type']
#             role_to_args[role].append(arg)
#         for role, role_args in role_to_args.items():
#             is_explicit = 'explicit' in (arg['arg_type'] for arg in role_args)
#             is_same_sent = 'same_sent' in (arg['position'] for arg in role_args)
#             arg_type = 'explicit' if is_explicit else 'implicit'
#             position = 'same_sent' if is_same_sent else 'cross_sent'
#             if not role_args:
#                 continue
#             gold_answers = [a['text'] for a in role_args]
#             gold_spans = [a['span'] for a in role_args]
#             questions = self.generator.generate(predicate_lemma, role)
#             for question in questions:
#                 yield {
#                     'doc_id': doc_id, 'sent_id': sent_id,
#                     'question': question, 'gold_answers': gold_answers,
#                     'predicate': pred_text,
#                     'predicate_lemma': predicate_lemma,
#                     'role_type': role,
#                     'text': frame['text'],
#                     'arg_type': arg_type, 'position': position,
#                     'gold_answer_spans': gold_spans, 'predicate_span': pred_span}
#
#     def load_srl_role_linking_as_qa(self, srl_path):
#         with jsonlines.open(srl_path) as fin:
#             frames = list(fin)
#         frames = [verify_sent_id(s) for s in frames]
#         if 'doc_id' not in frames[0] or 'sent_id' not in frames[0]:
#             raise ValueError("Verify that doc_id, sent_id are in the dataset")
#
#         all_qa_pairs = []
#         for frame in tqdm(frames, desc="Collecting questions"):
#             qa_pairs = list(self.yield_qa_inputs(frame))
#             for qa in qa_pairs:
#                 qa['gold_answer_spans'] = "~!~".join(qa['gold_answer_spans'])
#                 qa['gold_answers'] = "~!~".join(qa['gold_answers'])
#             all_qa_pairs.extend(qa_pairs)
#         return all_qa_pairs
#
#     @classmethod
#     def load_srl_template_qa_samples(cls, srl_path, question_path, proto_question_path):
#         generator = TemplateQuestionGenerator.from_path(question_path, proto_question_path)
#         qa_loader = cls(generator)
#         samples = qa_loader.load_srl_role_linking_as_qa(srl_path)
#         return samples
