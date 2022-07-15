from argparse import Namespace
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer

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


class QADiscourse_Pipeline(Text2TextGenerationPipeline):
    def __init__(self, model_repo: str, **kwargs):
        model, tokenizer = load_trained_model(model_repo)
        super().__init__(model, tokenizer, framework="pt")
        self.special_tokens = get_markers_for_model()
        self._update_config(**kwargs)
    
    def _update_config(self, **kwargs):
        " Update self.model.config with initialization parameters and necessary defaults. "
        # set default values that will always override model.config, but can overriden by __init__ kwargs
        kwargs["max_length"] = kwargs.get("max_length", 120)
        # override model.config with kwargs
        for k,v in kwargs.items():
            self.model.config.__dict__[k] = v       

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            processed_inputs = self._preprocess_string(inputs)
        elif hasattr(inputs, "__iter__"):
            processed_inputs = [self._preprocess_string(s) for s in inputs]
        else:
            raise ValueError("inputs must be str or Iterable[str]")
        # Now pass to super.preprocess for tokenization
        return super().preprocess(processed_inputs)
    
    def _preprocess_string(self, seq: str) -> str:
        seq = self.special_tokens.source_prefix + seq
        print(seq)
        return seq
    
    def _forward(self, *args, **kwargs):
        outputs = super()._forward(*args, **kwargs)
        return outputs


    def postprocess(self, model_outputs):
        predictions = self.tokenizer.decode(model_outputs["output_ids"].squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        seperated_qas = self._split_to_list(predictions)
        qas = []
        for qa_pair in seperated_qas:
            post_process = self._postrocess_qa(qa_pair) # if the prediction isn't a valid QA
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
        return {"question": question, "answer": answer}
    
    
if __name__ == "__main__":
    pipe = QADiscourse_Pipeline("RonEliav/QA_discourse")
    res1 = pipe("I don't like chocolate, but I like cookies.")
    res2 = pipe(["I don't like chocolate, but I like cookies.",
                 "I dived in the sea easily"], num_beams=10)
    print(res1)
    print(res2)