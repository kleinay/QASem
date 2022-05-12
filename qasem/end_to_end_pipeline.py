from typing import Iterable, Optional

# from qanom.nominalization_detector import NominalizationDetector
from qanom.nominalization_detector import NominalizationDetector
# from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline
from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline
import spacy
import nltk
from nltk.downloader import Downloader

import csv
import re
from collections import defaultdict
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from roleqgen.srl_as_qa_parser import PropBankRoleFinder
from roleqgen.question_translation import QuestionTranslator
from argparse import ArgumentParser
import huggingface_hub as HFhub

from spacy.tokenizer import Tokenizer
from typing import List

qanom_models = {"baseline": "kleinay/qanom-seq2seq-model-baseline",
                "joint": "kleinay/qanom-seq2seq-model-joint"}  

default_detection_threshold = 0.7
default_model = "joint"
default_annotation_layers = ['qanom', 'qasrl']
transformation_model_path = "biu-nlp/contextualizer_qasrl"
device_number = 0


# by default, use nltk's default pos_tagger ('averaged_perceptron_tagger'):
tagger_package = 'averaged_perceptron_tagger'
nltk_downloader = Downloader()
if not nltk_downloader.is_installed(tagger_package):
    nltk.download(tagger_package)
pos_tag = nltk.pos_tag


class QASemEndToEndPipeline():
    """
    This pipeline wraps QAnom pipeline and qasrl pipeline
    """
    def __init__(self,
                 qasrl_model: Optional[str] = None,  # for verbal and nominal verb
                 nominalization_detection_threshold: Optional[float] = None,
                 contextualize: bool = True,
                 annotation_layers: Optional[List[str]] = None
                 ):

        self.predicate_detector = NominalizationDetector()
        self.nominalization_detection_threshold = nominalization_detection_threshold or default_detection_threshold
        
        qanom_model = qasrl_model or default_model
        model_url = qanom_models[qanom_model] if qanom_model in qanom_models else qanom_model
        self.qa_pipeline = QASRL_Pipeline(model_url)

        self.contextualize = contextualize

        if self.contextualize:
            self.q_translator = QuestionTranslator.from_pretrained(transformation_model_path, device_id=int(device_number))

        self.annotation_layers = annotation_layers or default_annotation_layers


    def __call__(self, sentences: Iterable[str], 
                 nominalization_detection_threshold=None,
                 return_detection_probability=True,
                 **generate_kwargs):

        if 'qanom' in self.annotation_layers:
        # get predicates
            threshold = nominalization_detection_threshold or self.nominalization_detection_threshold
            predicate_infos_for_sentences = self.predicate_detector(sentences, 
                                                                    threshold=threshold,
                                                                    return_probability=return_detection_probability)

            inputs_model, input_sentence_index, input_verbs_form, pred_infos_input = self._prepare_input_sentences(sentences ,predicate_infos_for_sentences)
            model_output = self.qa_pipeline(inputs_model,
                                        verb_form=input_verbs_form,
                                        predicate_type="nominal", **generate_kwargs)
            if len(inputs_model) > 0:
                if self.contextualize:
                    # for contextualization
                    context_samples = []
                    self.contextual_qa(model_output, context_samples, pred_infos_input, sentences, input_sentence_index)

            outputs_nom = [[] for k in range(len(sentences))]
            # collect QAs for all predicates in sentence
            for model_pred_output, pred_info, sent_index in zip(model_output, pred_infos_input, input_sentence_index):

                predicates_full_info = dict(QAs=model_pred_output['QAs'], **pred_info)
                outputs_nom[sent_index].append(predicates_full_info)


        if 'qasrl' in self.annotation_layers:
            outputs_qasrl = [[] for k in range(len(sentences))]
            # qasrl detection
            # keep dictionary for all the verb in the sentence
            predicate_lists = [[]] * len(sentences) 
            nlp = spacy.load('en_core_web_sm')
            tokenizer = Tokenizer(nlp.vocab)
            nlp.tokenizer = tokenizer
            sentences_nlp = list(nlp.pipe(sentences))
            for i, sentence_nlp in enumerate(sentences_nlp):
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

            inputs_model, input_sentence_index, input_verbs_form, pred_infos_input = self._prepare_input_sentences(sentences, predicate_lists)
            model_output = self.qa_pipeline(inputs_model,
                                            verb_form=input_verbs_form,
                                            predicate_type="verbal",
                                            **generate_kwargs)
            if len(inputs_model) > 0:
                if self.contextualize:
                    # for contextualization
                    context_samples = []
                    self.contextual_qa(model_output, context_samples, pred_infos_input, sentences, input_sentence_index)

            # collect QAs for all predicates in sentence
            for model_pred_output, pred_info, sent_index in zip(model_output, pred_infos_input, input_sentence_index):
                predicates_full_info = dict(QAs=model_pred_output['QAs'], **pred_info)
                outputs_qasrl[sent_index].append(predicates_full_info)

        
        outputs = []
        for output_nom, output_qasrl in zip(outputs_nom, outputs_qasrl):
            outputs.append({ 
                'qanom': output_nom,
                'qasrl': output_qasrl
            })
    
        return outputs
    
    def _prepare_input_sentence(self, raw_sentence: str, predicate_idx: int) -> str:
        words = raw_sentence.split(" ") 
        words = words[:predicate_idx] + ["<predicate>"] + words[predicate_idx:] 
        return " ".join(words)

    def _prepare_input_sentences(self, raw_sentences, predicate_infos_for_sentences):
        sentences_input = []
        input_sentence_index = []
        input_verbs_form = []
        pred_infos_flatten = []
        index_sentence = 0
        for sentence, predicate_infos in zip(raw_sentences, predicate_infos_for_sentences):
            model_input = [self._prepare_input_sentence(sentence, pred_info['predicate_idx']) for pred_info in predicate_infos]
            verbs_form = [pred_info['verb_form'] for pred_info in predicate_infos]
            pred_infos = [pred_info for pred_info in predicate_infos]
            # sentences_input.append(model_input)
            sentences_input.extend(model_input)
            input_verbs_form.extend(verbs_form)
            pred_infos_flatten.extend(pred_infos)
            input_sentence_index.extend([index_sentence]*len(model_input))
            index_sentence += 1
        return sentences_input, input_sentence_index, input_verbs_form, pred_infos_flatten



    def contextual_qa(self,model_output, context_samples, pred_infos, sentences, sentences_index):
        for pred_output, pred_info, sentence_index in zip(model_output, pred_infos, sentences_index):
            for qa in pred_output['QAs']:
                context_samples.append({'proto_question': qa['question'], 'predicate_lemma': pred_info['verb_form'],
                                    'predicate_span': str(pred_info['predicate_idx']) + ':' + str(
                                        pred_info['predicate_idx'] + 1),
                                    'text': sentences[sentence_index]})

        contextualized_questions = self.q_translator.predict(context_samples)

        i = 0
        for pred_output, sentence_index in zip(model_output, sentences_index):
            for qa in pred_output['QAs']:
                qa['contextual_question'] = contextualized_questions[i]
                i += 1

if __name__ == "__main__":
    pipe = QASemEndToEndPipeline(nominalization_detection_threshold=0.75)
    sentences = ["The doctor was interested in Luke 's treatment .", "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."] #, "The Veterinary student was interested in Luke 's treatment of sea animals .", "Tom brings the dog to the park."]
    outputs = pipe(sentences, return_detection_probability=True)
    #
    print(outputs)

    # pipe = QASemEndToEndPipeline(detection_threshold=0.75)
    sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
    print(pipe([sentence]))
    # print()
    res1 = pipe(["The student was interested in Luke 's research about see animals ."])
    print(res1)
    print()#, verb_form="research", predicate_type="nominal")
    res2 = pipe(["The doctor was interested in Luke 's treatment .", "The Veterinary student was interested in Luke 's treatment of sea animals ."])#, verb_form="treat", predicate_type="nominal", num_beams=10)
    print(res2)
    # # #res3 = pipe(["A number of professions have developed that specialize in the treatment of mental disorders ."])
    # # # print(res1)
    #print(res2)
    print('\n')
    res3 = pipe(['Tom brings the dog to the park.'])
    print(res3)


