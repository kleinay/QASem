from argparse import Namespace
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from qasem.candidates_finder import CandidateFinder

def get_markers_for_model():
    special_tokens_constants = Namespace()
    special_tokens_constants.separator_different_qa = "&&&"
    special_tokens_constants.separator_output_question_answer = "SSEEPP"
    special_tokens_constants.source_prefix = "qa: "
    return special_tokens_constants

def load_trained_model(name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    return model, tokenizer

def lexical_coverage_ratio(str1: str, str2: str) -> float:
    """
    Return how many of `str1` words are covered by `str2` / num words in `str1`.
    Use simple `split()` for tokenization.
    """
    words1 = str1.split()
    words2 = str2.split()
    covered = [w for w in words1 if w in words2]
    return len(covered) / len(words1)

class QADiscourse_Pipeline(Text2TextGenerationPipeline):
    def __init__(self, model_repo: str, device=-1, **kwargs):
        " :param device: -1 for CPU (default), >=0 refers to CUDA device ordinal. "
        model, tokenizer = load_trained_model(model_repo)
        super().__init__(model, tokenizer, device=device, framework="pt")
        self.special_tokens = get_markers_for_model()
        self._update_config(**kwargs)
        self.cand_finder = CandidateFinder()

    def _update_config(self, **kwargs):
        " Update self.model.config with initialization parameters and necessary defaults. "
        # set default values that will always override model.config, but can overriden by __init__ kwargs
        kwargs["max_length"] = kwargs.get("max_length", 120)
        # override model.config with kwargs
        for k, v in kwargs.items():
            self.model.config.__dict__[k] = v

    def preprocess(self, inputs):
        # only forward sentences with +2 predicates / clauses
        if self.cand_finder.num_candidates(inputs) < 2:
            processed_inputs = ""  # There in not dicourse relation in this sentence, pass null to model
        else:
            processed_inputs = self._preprocess_string(inputs)
        # Now pass to super.preprocess for tokenization
        return super().preprocess(processed_inputs)

    def _preprocess_string(self, seq: str) -> str:
        seq = self.special_tokens.source_prefix + seq
        return seq

    def _forward(self, *args, **kwargs):
        outputs = super()._forward(*args, **kwargs)
        return outputs

    def postprocess(self, model_outputs):
        predictions = self.tokenizer.decode(model_outputs["output_ids"].squeeze(), skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        seperated_qas = self._split_to_list(predictions)
        qas = []
        for qa_pair in seperated_qas:
            post_process = self._postrocess_qa(qa_pair)  # if the prediction isn't a valid QA
            if post_process is not None:
                qas.append(post_process)
        return qas

    def _split_to_list(self, output_seq: str) -> list:
        return list(set(output_seq.split(self.special_tokens.separator_different_qa)))

    def _postrocess_qa(self, seq: str) -> str:
        # split question and answers
        if self.special_tokens.separator_output_question_answer in seq:
            question, answer = seq.split(self.special_tokens.separator_output_question_answer)
        else:
            return None
        # Heuristic filters applied due to model over-generation -
        #   filter answers that are lexically covered by the question
        if lexical_coverage_ratio(answer, question) >= 0.75:
            return None
        #   filter question that are lexically covered (excluding prefix) by the answer
        if lexical_coverage_ratio(' '.join(question.split()[4:]), answer) >= 0.8:
            return None
        return {"question": question, "answer": answer}


if __name__ == "__main__":
    import sys
    pipe = QADiscourse_Pipeline("RonEliav/QA_discourse_v2")
    if len(sys.argv)==1:
        res1 = pipe("I don't like chocolate, but I like cookies.")
        res2 = pipe(["I don't like chocolate, but I like cookies.",
                     "I am George.",
                     "I left school a several years ago."], num_beams=3)
        print(res1)
        print(res2)
    else:
        res = pipe(sys.argv[1:], num_beams=3)
        print(res)