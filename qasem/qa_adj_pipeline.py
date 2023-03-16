import itertools
from typing import Optional, List, Dict, Union
from argparse import Namespace
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

default_model_name = "leonpes/qaadj_parser"
default_tokenizer_name = "t5-base"

class QAAdj_Pipeline(Text2TextGenerationPipeline):
    """ 
    Given and adjectival predicate, predicts QAs corresponding to the Object, Domain, Reference, and Degree of the adjective.
    The adjectival predicate is assumed to be marked using a [PRED] marker before and after the predicate (spaced),
    e.g. "A [PRED] female [PRED] teacher...". 
    """
    ROLES = ('object', 'comparison', 'domain', 'extent')
    def __init__(self, model_repo: Optional[str] = None, device=-1, **kwargs):
        " :param device: -1 for CPU (default), >=0 refers to CUDA device ordinal. "
        if model_repo is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(default_model_name)
            tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
            tokenizer = AutoTokenizer.from_pretrained(model_repo)
        super().__init__(model, tokenizer, device=device, framework="pt")
        self.tokenizer.add_tokens(['[QASEP]','[NOQA]', '[PRED]'])
        # self.model.resize_token_embeddings(len(self.tokenizer))   # I didn't test whether it is better to include this
        self._update_config(**kwargs)

    def _update_config(self, **kwargs):
        " Update self.model.config with initialization parameters and necessary defaults. "
        # set default values that will always override model.config, but can overriden by __init__ kwargs
        kwargs["max_length"] = kwargs.get("max_length", 100)
        kwargs["num_beams"] = kwargs.get("num_beams", 3)
        kwargs["repetition_penalty"] = kwargs.get("repetition_penalty", 3.5)
        # override model.config with kwargs
        for k, v in kwargs.items():
            self.model.config.__dict__[k] = v

    def preprocess(self, inputs):
        # Here, inputs is string or list of strings; apply string postprocessing
        if isinstance(inputs, str):
            processed_inputs: List[str] = self._preprocess_string(inputs)
        elif hasattr(inputs, "__iter__"):
            processed_inputs: List[List[str]] = [self._preprocess_string(s) for s in inputs]
            processed_inputs = list(itertools.chain(*processed_inputs)) # flat
        else:
            raise ValueError("inputs must be str or Iterable[str]")
        # Now pass to super.preprocess for tokenization
        return super().preprocess(processed_inputs)

    def _preprocess_string(self, seq: str) -> List[str]:
        """ For each instance (adjective in sentence), return 4 sequences, to run the model 4 times, once per role."""
        seqs = [f"{role}: Sentence: {seq}" 
                for role in QAAdj_Pipeline.ROLES]
        return seqs

    def _forward(self, *args, **kwargs):
        outputs = super()._forward(*args, **kwargs)
        return outputs

    def postprocess(self, model_outputs):
        # TODO solve this complexity - postprocess presumable will be called for each "forward" advocation,
        # but we want to aggregate 4 subsequent calls (one per role). How to do this?
        # Can we control for the "BATCH" axis to get (duplicates of) 4 in each call?
        predictions = self.tokenizer.batch_decode(model_outputs["output_ids"].squeeze(), 
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        assert len(predictions) == 4, "Each instance should postprocess a batch of 4 QAs (one per role)"
        def decode_qa(qa_string) -> Optional[Union[str, Dict[str, str]]]:
            if qa_string == "[NOQA]":
                return None
            elif "?" not in qa_string:
                return qa_string
            question, answer = qa_string.split('?')
            question = question[9:].strip() + '?'
            answer = answer[8:].strip()
            return {"question": question, "answer": answer}
        
        out = {role: decode_qa(predicted_qa)
               for role, predicted_qa in zip(QAAdj_Pipeline.ROLES, predictions)}
        # out = {"question": question, "answer": answer}
        return out


if __name__ == "__main__":
    import sys
    pipe = QAAdj_Pipeline()
    if len(sys.argv)==1:
        # res1 = pipe("I don't like very [PRED] brown [PRED] chocolate, but I like cookies .")
        res2 = pipe(["I don't like very [PRED] brown [PRED] chocolate, but I like cookies .",
                     "George is [PRED] good [PRED] in basketball for his age .",
                     "John is [PRED] better [PRED] at Tennis than Billy .",
                     "I am [PRED] sensitive [PRED] to these tricks ."], num_beams=3)
        # print(res1)
        print(res2)
    else:
        res = pipe(sys.argv[1:], num_beams=3)
        print(res)