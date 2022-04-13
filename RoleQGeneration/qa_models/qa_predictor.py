import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForQuestionAnswering, AutoTokenizer

from .qa_dataset import QuestionAnswerDataset
from .qa_model import QuestionAnswerModule


class QuestionAnswerPredictor:
    def __init__(self, qa_model: PreTrainedModel, qa_tokenizer: PreTrainedTokenizer, device: torch.device):
        self.qa_model = qa_model
        self.tokenizer = qa_tokenizer
        self.qa_infer = QuestionAnswerModule(self.qa_model, sep_token_id=self.tokenizer.sep_token_id)
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_path: str, device_id=-1):
        device = torch.device("cpu")
        if device_id != -1 and torch.cuda.is_available():
            device = torch.device(device_id)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_path).to(device)
        qa_tok = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
        return cls(qa_model, qa_tok, device)

    def predict(self, question, text):
        # We could also easily change this to support batch prediction!
        sample = {"question": question, "text": text}
        dataset = QuestionAnswerDataset([], self.tokenizer,
                                        use_predicate_indicator=False,
                                        max_length=384)
        samples, batch = dataset.tokenize_collate_with_samples([sample])
        infer_spans = self.qa_infer.infer(samples, batch, self.device)
        return infer_spans[0]
