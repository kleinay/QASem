import os
from argparse import ArgumentParser
import gzip
from typing import Any, Dict
import jsonlines
from tqdm import tqdm

from qa_models import QuestionAnswerDataset


def is_reliable_question(question_entry):
    judgments = [judge['isValid'] for judge in question_entry['answerJudgments']]
    n_valid = sum(judgments)
    # The human question generator is added to the list of judges.
    n_humans = len(judgments)
    valid_ratio = float(n_valid) / n_humans
    # at least three humans thought this is valid (3/3 for the regular training data)
    # and at least 5/5 humans (for model generated questions in dense) or 5/6 humans thought this is valid
    is_reliable = n_valid > 2 and valid_ratio > 0.8
    return is_reliable


def get_answer_spans(question_entry, tokens):
    candidate_answer_spans = [judge['spans'] for judge in question_entry['answerJudgments'] if judge['isValid']]
    candidate_answer_spans = [tuple(span) for spans in candidate_answer_spans for span in spans]
    # remove duplicate answers
    candidate_answer_spans = sorted(set(candidate_answer_spans))
    candidate_answers = [" ".join(tokens[start: end]) for start, end in candidate_answer_spans]
    candidate_answer_spans = [f"{start}:{end}" for start, end in candidate_answer_spans]
    candidate_answer_spans = "~!~".join(candidate_answer_spans)
    candidate_answers = "~!~".join(candidate_answers)
    return candidate_answer_spans, candidate_answers


def yield_questions(verb_entry, tokens):
    pred_idx = verb_entry['verbIndex']
    pred_span = f"{pred_idx}:{pred_idx + 1}"
    pred_text = tokens[pred_idx]
    for question_entry in verb_entry['questionLabels'].values():
        if not is_reliable_question(question_entry):
            continue
        question = question_entry['questionString']
        answer_spans, answers = get_answer_spans(question_entry, tokens)
        record = {
            'question': question,
            'gold_answers': answers,
            'gold_answer_spans': answer_spans,
            'predicate_span': pred_span,
            'predicate': pred_text,
            'role_type': "UNK",
            'is_negated': question_entry['isNegated'],
            'is_passive': question_entry['isPassive'],
        }
        record.update(question_entry['questionSlots'])
        yield record


def parse_sent_id(sent_id):
    if sent_id.startswith("Wiki"):
        doc_id, sent_idx = sent_id.rsplit(":", maxsplit=1)
    elif sent_id.startswith("TQA"):
        doc_id, sent_idx = sent_id.rsplit("_", maxsplit=1)
    else:
        doc_id, sent_idx = "UNK", -1
    sent_idx = int(sent_idx)

    return doc_id, sent_idx


def yield_records_from_qasrl(qasrl_sent: Dict[str, Any]):
    # {"sentenceId": "TQA:T_0058_1", "sentenceTokens": ["X",...],
    #  "verbEntries": {
    #      "1": {"verbIndex": 1,
    #           "verbInflectedForms": {...},
    #           "questionLabels": { "What occurs?":
    #               {"questionString": "What occurs?",
    #                "questionSources": ["turk-qasrl2.0-1328"],
    #                "answerJudgments": [ {"sourceId": "turk-qasrl2.0-1153", "isValid": true, "spans": [[0, 1]]},
    #                                     {"sourceId": "turk-qasrl2.0-1367", "isValid": true, "spans": [[0, 1]]},
    #                                     {"sourceId": "turk-qasrl2.0-1328", "isValid": true, "spans": [[0, 1]]}],
    #                 "questionSlots": {"wh": "what", "aux": "_", "subj": "_", "verb": "presentSingular3rd",
                                 #                   "obj": "_", "prep": "_", "obj2": "_"}, "tense": "present",
                                 #      "isPerfect": false, "isProgressive": false, "isNegated": false, "isPassive": false},...}

    doc_id, sent_idx = parse_sent_id(qasrl_sent['sentenceId'])
    tokens = qasrl_sent['sentenceTokens']
    text = " ".join(tokens)
    for verb_entry in qasrl_sent['verbEntries'].values():
        for rec in yield_questions(verb_entry, tokens):
            rec['doc_id'] = doc_id
            rec['sent_id'] = sent_idx
            rec['role_type'] = "UNK"
            rec['arg_type'] = "UNK"
            rec['text'] = text
            yield rec


def main(args):
    for part in ('train', 'dev'):
        qasrl_data_path = os.path.join(args.qasrl_path, "expanded", f"{part}.jsonl.gz")
        print(f"Reading from {qasrl_data_path}")
        all_records = []
        with gzip.open(qasrl_data_path) as gz_in:
            with jsonlines.Reader(gz_in) as qasrl_reader:
                for qasrl_sent in tqdm(qasrl_reader, desc="Parsing QA-SRL"):
                    for rec in yield_records_from_qasrl(qasrl_sent):
                        all_records.append(rec)
        # if args.with_long_context:
        #     all_records = add_long_context(all_records)

        # cols = ['doc_id', 'sent_id', 'predicate', 'role_type',
        #         'question', 'gold_answers', 'arg_type',
        #         'gold_answer_spans', 'predicate_span', 'text']
        out_path = args.out_path.format(part=part)
        QuestionAnswerDataset.save_tsv_samples(all_records, out_path)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--qasrl_path", help="C:\\dev\\qasrl-v2")
    ap.add_argument("--out_path", help="./qasrl/qasrl.expanded.{part}.tsv")
    # ap.add_argument("with_long_context", action="store_true")
    main(ap.parse_args())
