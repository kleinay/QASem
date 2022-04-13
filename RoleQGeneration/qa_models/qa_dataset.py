import os
from typing import Dict, Any, List

from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizerBase
import logging
from unidecode import unidecode

# from .dataloaders import load_tsv_samples, load_mrqa_samples, SrlRoleLinkingLoader
from .dataloaders import load_tsv_samples, load_mrqa_samples
from .qa_utils import \
    surround_predicate_with_indicators, \
    pick_random_answer, \
    batch_translate_token_subword_positions

logger = logging.getLogger(__name__)

# The questions tokens usually have 0, and the context have 1.
# To indicate inside the context that this token is a predicate token use the 0 type.
# This might confuse the model at the beginning..
# but we don't have to change the token_type embedding matrix
PREDICATE_INDICATOR_TOKEN_TYPE = 0


def assert_batch_matches_sample(batch_inputs, new_samples):
    for sidx, sample in enumerate(new_samples):
        tokens = sample['text'].split()
        gold_answer = ''.join(tokens[sample['answer_start']: sample['answer_end']]).replace("#", '').lower()
        subwords = batch_inputs.encodings[sidx].tokens
        sub_start = batch_inputs['start_positions'][sidx].item()
        sub_end = batch_inputs['end_positions'][sidx].item()
        expected_answer = ''.join(subwords[sub_start:(sub_end + 1)]).replace("#", '').lower()
        if unidecode(gold_answer) != unidecode(expected_answer):
            print(f"{gold_answer}    *******    {expected_answer}")


class QuestionAnswerDataset(Dataset):
    PREP_ARGS = {
        'max_length': 128,
        'add_special_tokens': True,
        'padding': 'max_length',
        'return_tensors': 'pt',
        'return_offsets_mapping': True,
        'truncation': True
    }

    def __init__(self, samples, tokenizer: PreTrainedTokenizerBase, use_predicate_indicator=True, **kwargs):
        """

        :param samples:
        :param tokenizer:
        :param kwargs:
        """
        self.tokenizer = tokenizer
        self.max_length = kwargs['max_length']
        self.samples = samples
        # counts edge cases when the answer appears after the max desired length
        self.n_answer_out_of_scope = 0
        tokenizer_args = dict(self.PREP_ARGS)
        tokenizer_args.update(kwargs)
        self.tokenizer_args = tokenizer_args
        self.use_predicate_indicator = use_predicate_indicator

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def tokenize_collate_with_samples(self, samples: List[Dict[str, Any]]):
        # --------------------------------
        # COMMENT out dynamic-max-length because our expected sub-word length
        # calculation may not be accurate enough.
        # Let's see if this solves some bug for us.
        #
        # dynamic_max_length = _find_dynamic_max_length(questions, texts)
        # hard_max_length = self.tokenizer_args['max_length']
        # Save a few kw by not doing gigantic computations on [PAD] tokens?
        # Or am I being overly naive..
        # dynamic_max_length = min(dynamic_max_length, hard_max_length)
        # dynamic_args = dict(self.tokenizer_args)
        # dynamic_args['max_length'] = dynamic_max_length
        # --------------------------------

        new_samples = samples
        # remove consecutive whitespaces, otherwise our alignment through character indices
        # between subwords and original tokens will not work properly.
        # BERT tokenizer maintains the original character indices
        # while we get a list of indices after naive text.split() on whitespace
        for s in samples:
            s['text'] = " ".join(s['text'].split())

        if self.use_predicate_indicator:
            new_samples = [surround_predicate_with_indicators(s) for s in samples]
        new_samples = [pick_random_answer(s) for s in new_samples]
        questions = [s['question'] for s in new_samples]
        texts = [s['text'] for s in new_samples]
        batch_inputs = self.tokenizer(questions, texts, **self.tokenizer_args)

        new_samples, batch_inputs, n_not_ok = batch_translate_token_subword_positions(new_samples,
                                                                                      batch_inputs,
                                                                                      self.max_length)
        self.n_answer_out_of_scope += n_not_ok

        # USE THIS TO VERIFY YOUR INPUTS DURING DEBUGGING
        # BertTokenizer replaces unknown characters with UNK
        # and the batch may point to CLS if the answer is beyond
        # maximal length.
        # assert_batch_matches_sample(batch_inputs, new_samples)

        return new_samples, batch_inputs

    def tokenize_collate(self, samples: List[Dict[str, Any]]):
        _, batch_inputs = self.tokenize_collate_with_samples(samples)
        # PY-TORCH LIGHTNING ON DistributedParallel (and DistributedDataParallel)  calls torch.scatter
        # with batch_inputs: BatchEncoding. However, this is not recognized as a dictionary,
        # and is not scattered correctly.
        batch_inputs_as_dict = dict(batch_inputs)
        # Add the sub-words for debugging
        batch_inputs_as_dict['subwords'] = [enc.tokens for enc in batch_inputs.encodings]
        return batch_inputs_as_dict

    @classmethod
    def save_tsv_samples(cls, samples, out_path, required_cols=None):
        for s in samples:
            answer_spans = s.get('gold_answer_spans', [])
            if isinstance(answer_spans, list):
                new_spans = []
                for start, end in answer_spans:
                    new_spans.append(f"{start}:{end}")
                s['gold_answer_spans'] = "~!~".join(new_spans)
            answers = s.get('gold_answers', [])
            if isinstance(answers, list):
                s['gold_answers'] = "~!~".join(answers)
            predicate_span = s.get('predicate_span')
            if isinstance(predicate_span, tuple):
                s['predicate_span'] = f"{predicate_span[0]}:{predicate_span[1]}"

        df = pd.DataFrame(samples)
        if not required_cols:
            required_cols = df.columns.tolist()
        existing_cols = set(samples[0].keys())
        cols = [c for c in required_cols if c in existing_cols]
        df[cols].to_csv(out_path, index=False, encoding='utf-8', sep="\t")

    @classmethod
    def load_samples(cls, dataset_path, fast_dev_run=False, **kwargs):
        ext = os.path.splitext(dataset_path)[1]

        if ext == '.gz':
            samples = load_mrqa_samples(dataset_path, fast_dev_run)
        elif ext in (".tsv", ".csv"):
            samples = load_tsv_samples(dataset_path, fast_dev_run)
        # elif ext == ".jsonl":
            # samples = SrlRoleLinkingLoader.load_srl_template_qa_samples(dataset_path, **kwargs)
        else:
            raise ValueError("Which loading function to use?")
        return samples

    @classmethod
    def load_dataset(cls, dataset_path, tokenizer: PreTrainedTokenizerBase,
                     question_path=None,
                     fast_dev_run=False, **kwargs):
        samples = cls.load_samples(dataset_path, fast_dev_run)
        return cls(samples, tokenizer, **kwargs)
