from typing import Any, Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, BartForConditionalGeneration, BartTokenizer


MAX_QUESTION_LENGTH = 30
MAX_TEXT_LENGTH = 192
# For adjuncts.
PREDICATE_PLACEHOLDER = "<PLACEHOLDER>"


TOK_ARGS = {
    "return_tensors": "pt",
    "max_length": MAX_TEXT_LENGTH,
    "padding": True,
    "truncation": True
}


class QuestionContextualizer:
    def __init__(self, gen_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device):
        self.gen_model = gen_model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_path, device_id=-1):
        tok = BartTokenizer.from_pretrained(pretrained_path)
        device = torch.device("cpu")
        if device_id != -1 and torch.cuda.is_available():
            device = torch.device(device_id)
        gen_model = BartForConditionalGeneration.from_pretrained(pretrained_path).to(device)
        return cls(gen_model, tok, device)

    def predict(self, samples: List[Dict[str, Any]]):
        is_batch = True
        if isinstance(samples, dict):
            is_batch = False
            samples = [samples]

        gen_args = {"num_beams": 1, "max_length": MAX_QUESTION_LENGTH, "early_stopping": True}
        inputs = [self._to_text_input(s["proto_question"],
                                      s["predicate_lemma"],
                                      s["predicate_span"],
                                      s["text"])
                  for s in samples]
        batch_encodings = self.tokenizer(inputs, **TOK_ARGS).to(self.device)
        batch_encodings = dict(batch_encodings)
        batch_encodings.update(gen_args)
        summary_ids = self.gen_model.generate(**batch_encodings).detach().cpu()
        questions = []
        for sum_ids in summary_ids:
            question = self.tokenizer.decode(sum_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            # Funny generation artifacts after the first question mark (?)
            question = question.split('?')[0] + '?'
            questions.append(question)
        if not is_batch:
            return questions[0]
        return questions

    @staticmethod
    def _to_text_input(proto_question, predicate_lemma, predicate_span, text):
        tokens = text.split()
        pred_start, pred_end = parse_span(predicate_span)
        # probably should use verb form and not predicate lemma...
        if PREDICATE_PLACEHOLDER in proto_question:
            proto_question = proto_question.replace(PREDICATE_PLACEHOLDER, predicate_lemma)
        # MADNESS. Valentina change the separators.
        # You can PREDICATE_START see PREDICATE_END an example . </s> see [SEP] where is something seen ?
        text_input = f"{' '.join(tokens[:pred_start])}" \
                     f" PREDICATE_START {' '.join(tokens[pred_start:pred_end])} PREDICATE_END " \
                     f"{' '.join(tokens[pred_end:])} </s> " \
                     f"{predicate_lemma} [SEP] {proto_question}"
        return text_input

def parse_span(span: str):
    start, end = span.split(":")
    return int(start), int(end)
