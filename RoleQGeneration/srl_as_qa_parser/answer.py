from typing import List, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForQuestionAnswering

from RoleQGeneration.qa_models import QuestionAnswerDataset, QuestionAnswerModule
from RoleQGeneration.qa_models.qa_utils import batch_remove_predicate_indicators

Span = Tuple[int, int]


class HuggingFaceQAPredictor:
    def __init__(self, qa_model: nn.Module, tokenizer: PreTrainedTokenizerBase,
                 max_length=512,
                 decode_mode=QuestionAnswerModule.DECODE_MODE_ARGMAX_GREEDY):
        self.qa_model = qa_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.decode_mode = decode_mode

    def predict(self, questions: Union[str, List[str]], tokens: List[str], pred_span: Tuple[int, int], search_space) -> Span:
        if isinstance(questions, str):
            questions = [questions]

        records = [{'text': " ".join(tokens),
                    'question': question,
                    'search_space': search_space,
                    "predicate_span": pred_span} for question in questions]
        # This is not intended to be used with a data-loader with division to batches.
        # Supply only several questions (from the same role possibly)
        dataset = QuestionAnswerDataset([], tokenizer=self.tokenizer, max_length=self.max_length)
        tokenized_records, batch = dataset.tokenize_collate_with_samples(records)
        qa_module = QuestionAnswerModule(self.qa_model, decode_mode=self.decode_mode)
        qa_module.to(self.device)
        spans = qa_module.infer(tokenized_records, batch, self.device)
        _, spans = batch_remove_predicate_indicators(tokenized_records, spans)
        # TODO: Why always a single span?
        return spans[0]

    @classmethod
    def from_path(cls, qa_model_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(qa_model_path, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)
        return HuggingFaceQAPredictor(model, tokenizer, **kwargs)
