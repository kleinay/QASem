from typing import Iterable, Optional

# from qanom.nominalization_detector import NominalizationDetector
from nominalization_detector import NominalizationDetector
# from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline
from qasrl_seq2seq_pipeline import QASRL_Pipeline
import spacy

qanom_models = {"baseline": "kleinay/qanom-seq2seq-model-baseline",
                "joint": "kleinay/qanom-seq2seq-model-joint"}  

default_detection_threshold = 0.7
default_model = "joint"


class QASemEndToEndPipeline():
    """
    This pipeline wraps QAnom pipeline and qasrl pipeline
    """
    def __init__(self, 
                 qanom_model: Optional[str] = None, 
                 detection_threshold: Optional[float] = None):
        self.predicate_detector = NominalizationDetector()
        self.detection_threshold = detection_threshold or default_detection_threshold
        
        qanom_model = qanom_model or default_model
        model_url = qanom_models[qanom_model] if qanom_model in qanom_models else qanom_model
        self.qa_pipeline = QASRL_Pipeline(model_url)

    def __call__(self, sentences: Iterable[str], 
                 detection_threshold = None,
                 return_detection_probability = True,
                 **generate_kwargs):
        # get predicates
        threshold = detection_threshold or self.detection_threshold
        predicate_infos_for_sentences = self.predicate_detector(sentences, 
                                                                threshold=threshold,
                                                                return_probability=return_detection_probability)
        outputs = []
        for sentence, predicate_infos in zip(sentences, predicate_infos_for_sentences):
            # collect QAs for all predicates in sentence 
            predicates_full_infos = []
            for pred_info in predicate_infos:
                model_input = self._prepare_input_sentence(sentence, pred_info['predicate_idx'])
                model_output = self.qa_pipeline(model_input, 
                                                verb_form=pred_info['verb_form'], 
                                                predicate_type="nominal",
                                                **generate_kwargs)[0]
                predicates_full_info = dict(QAs=model_output['QAs'], **pred_info)
                predicates_full_infos.append(predicates_full_info)
            outputs.append(predicates_full_infos)
        
        # qasrl detection
        # keep dictionary for all the verb in the sentence
        predicate_lists = [[]] * len(sentences) 
        nlp = spacy.load('en_core_web_sm')
        for i, sentence in enumerate(sentences):
            sentence_nlp = nlp(sentence)
            target_idxs = []
            verb_forms = []
            tokens_text = [w.text for w in sentence_nlp]
            for j, token in enumerate(sentence_nlp):
                if token.pos_ == 'VERB':
                    target_idxs.append(j)
                    verb_forms.append(token.lemma_)

            predicate_lists[i] = [
            {"predicate_idx": pred_idx,
                "predicate": tokens_text[pred_idx],
                "verb_form": verb_form} 
            for pred_idx, verb_form in zip(target_idxs, verb_forms)]

        outputs_qasrl = []
        for sentence, predicate_infos in zip(sentences, predicate_lists):
            # collect QAs for all predicates in sentence 
            predicates_full_infos = []
            for pred_info in predicate_infos:
                model_input = self._prepare_input_sentence(sentence, pred_info['predicate_idx'])
                model_output = self.qa_pipeline(model_input, 
                                                verb_form=pred_info['verb_form'], 
                                                predicate_type="verbal",
                                                **generate_kwargs)[0]
                predicates_full_info = dict(QAs=model_output['QAs'], **pred_info)
                predicates_full_infos.append(predicates_full_info)
            outputs_qasrl.append(predicates_full_infos)
        
        for i, output_nom in enumerate(outputs):
            output_nom.extend(outputs_qasrl[i])
    
        return outputs
    
    def _prepare_input_sentence(self, raw_sentence: str, predicate_idx: int) -> str:
        words = raw_sentence.split(" ") 
        words = words[:predicate_idx] + ["<predicate>"] + words[predicate_idx:] 
        return " ".join(words)



if __name__ == "__main__":
    pipe = QASemEndToEndPipeline(detection_threshold=0.75)
    # sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
    # print(pipe([sentence])) 
    #res1 = pipe(["The student was interested in Luke 's research about see animals ."])#, verb_form="research", predicate_type="nominal")
    res2 = pipe(["The doctor was interested in Luke 's treatment .", "The Veterinary student was interested in Luke 's treatment of sea animals .", "I eat a peach"])#, verb_form="treat", predicate_type="nominal", num_beams=10)
    #res3 = pipe(["A number of professions have developed that specialize in the treatment of mental disorders ."])
    # print(res1)
    print(res2)
    # print(res3)   

