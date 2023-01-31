import sys
from typing import Iterable, Optional, Tuple, List, Any, Dict
from qanom.nominalization_detector import NominalizationDetector
from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline
import spacy
import nltk
from tqdm import tqdm
from nltk.downloader import Downloader
from qasem.question_contextualizer import QuestionContextualizer
from spacy.tokenizer import Tokenizer
from qasem.qa_discourse_pipeline import QADiscourse_Pipeline
from qasem.openie_converter import OpenIEConverter
from qasem.utils import ListDataset

qasrl_models = {"baseline": "kleinay/qasrl-seq2seq-model",
                "joint": "kleinay/qanom-seq2seq-model-joint"}
qanom_models = {"baseline": "kleinay/qanom-seq2seq-model-baseline",
                "joint": "kleinay/qanom-seq2seq-model-joint",
                "order-invariant": "kleinay/qanom-seq2seq-model-order-invariant"}
default_qasrl_model = "joint"
default_qanom_model = "joint"
qadiscourse_model_name = "RonEliav/QA_discourse_v2"
question_contextualization_model_name = "biu-nlp/contextualizer_qasrl"

# Defaults
default_annotation_layers = ['qanom', 'qasrl', 'qadiscourse']
default_nominalization_detection_threshold = 0.7
NO_REPEAT = True
DEFAULT_SPACY_MODEL = 'en_core_web_sm'

class QASemEndToEndPipeline():
    """
    This pipeline currently wraps QA-SRL, QANom and QADiscourse pipelines.
    """
    def __init__(self,
                 annotation_layers: Optional[List[str]] = None,
                 device: int = -1, 
                 qasrl_model: Optional[str] = None,  # for verbal predicates
                 qanom_model: Optional[str] = None,  # for nominal predicates
                 nominalization_detection_threshold: Optional[float] = None,
                 contextualize: bool = False,
                 return_qasrl_slots: bool = False,
                 return_qasrl_discrete_role: bool = True,
                 spacy_model: Optional[str] = None,
                 qasrl_pipeline_kwargs = dict(),
                 openie_converter_kwargs: Dict[str, Any] = dict(),
                 ):
        """

        Args:
            annotation_layers (Optional[List[str]], optional): which QA-based semantic tasks should the output include. 
                Default includes all available layers (currently "qasrl", "qanom", "qadiscourse").
            device (int, optional): -1 for CPU (default), >=0 refers to CUDA device ordinal. Defaults to -1.
            qasrl_model (Optional[str], optional): Underlying verbal QASRL model. Can be a key for `qasrl_models` or a Huggingface Hub URL. 
                Defaults to "joint".
            qanom_model (Optional[str], optional): Underlying nominal QASRL (=QANom) model. Can be a key for `qanom_models` or a Huggingface Hub URL.
                Defaults to "joint".
            contextualize (bool, optional): . 
            openie_converter_kwargs (Dict[str, Any], optional): key-word args to pass to `OpenIEConverter` constructor. Defaults to empty dict().
        """
        self.device_int = device # represent device in HF convention
        self.device_str = f"cuda:{device}" if device >= 0 else "cpu"# represent device in pytorch convention
        self.annotation_layers = annotation_layers or default_annotation_layers
        qasrl_model = qasrl_model or default_qasrl_model
        qanom_model = qanom_model or default_qanom_model
        # Either a name from dict or the actual model name in HF
        qasrl_model_url = qasrl_models[qasrl_model] if qasrl_model in qasrl_models else qasrl_model
        qanom_model_url = qanom_models[qanom_model] if qanom_model in qanom_models else qanom_model
        # Init QANom predicate detection model
        if 'qanom' in self.annotation_layers:
            self.nominal_predicate_detector = NominalizationDetector(device=device)
            self.nominalization_detection_threshold = nominalization_detection_threshold or default_nominalization_detection_threshold

        # Set `self.qasrl_pipelines` for verbal and/or nominal QASRL
        qasrl_pipeline_kwargs = dict(return_question_slots=return_qasrl_slots,
                                     return_question_role=return_qasrl_discrete_role,
                                     device=device,
                                     **qasrl_pipeline_kwargs)
        
        if 'qasrl' in self.annotation_layers and 'qanom' in self.annotation_layers \
                and qasrl_model_url == qanom_model_url:
            # Default is using the same joint model for verbs and nominalizations (memory efficency)
            joint_pipe = QASRL_Pipeline(qasrl_model_url, **qasrl_pipeline_kwargs)
            self.qasrl_pipelines = {"verbal": joint_pipe, "nominal": joint_pipe}
        else:
            if 'qasrl' in self.annotation_layers:
                self.qasrl_pipelines = {"verbal": QASRL_Pipeline(qasrl_model_url, **qasrl_pipeline_kwargs)}
            if 'qanom' in self.annotation_layers:
                self.qasrl_pipelines = {"nominal": QASRL_Pipeline(qasrl_model_url, **qasrl_pipeline_kwargs)}

        if 'qadiscourse' in self.annotation_layers:
            self.qa_discourse_pipeline = QADiscourse_Pipeline(qadiscourse_model_name, device=device)

        self.contextualize = contextualize
        self.return_qasrl_slots = return_qasrl_slots
        self.return_qasrl_discrete_role = return_qasrl_discrete_role
        self.spacy_model_name = spacy_model or DEFAULT_SPACY_MODEL

        if self.contextualize:
            self.q_translator = QuestionContextualizer.from_pretrained(question_contextualization_model_name, device_id=device)

        self.openie_converter = OpenIEConverter(**openie_converter_kwargs)


    def __call__(self, sentences: Iterable[str], 
                 nominalization_detection_threshold: Optional[float] = None,
                 output_openie: bool = False,
                 verbose: bool = False,
                 **generate_kwargs):
        """
        By default, output would be a list in the same size as `sentences`,
         each item is a dictionary containing all the QAs of the sentence grouped by annotation-layer.

        If `output_openie`, output is in the form `{"qasem": qa_outputs, "openie": oie_outputs}`,
         where `qa_outputs` is the default QASem output list,
         and `oie_outputs` is also a `len(sentences)`-sized list, where each item is a list of OpenIE tuples
         pertaining to a sentence.

        Rest of keyword arguments (`**generate_kwargs`) are passed directly onto `model.generate`.
        """
        # Handle single sentence input
        if isinstance(sentences, str):
            res = self([sentences], 
                 nominalization_detection_threshold=nominalization_detection_threshold,
                 output_openie=output_openie,
                 **generate_kwargs
                 )
            return res[0] if isinstance(res, list) else res
        
        # POS-tag the sentences for extracting different predicate types 
        sentences_tokens_tags, sentences_pos, sentences_lemma = self.pos_tag_tokens(sentences)

        # Now handle each annotation layer one-by-one:
        
        outputs_nom = [[] for k in range(len(sentences))]
        if 'qanom' in self.annotation_layers:

            # get predicates
            threshold = nominalization_detection_threshold or self.nominalization_detection_threshold
            if verbose: print(f"Running QANom predicate detection...")
            predicate_lists = self.nominal_predicate_detector(sentences, pos_tagged_sentences=sentences_tokens_tags, threshold=threshold)
            # run QA generation model
            if verbose: print(f"Running QANom QA-generation...")
            outputs_nom = self.get_qasrl_qa(sentences, predicate_lists, 'nominal', **generate_kwargs)

        outputs_qasrl = [[] for k in range(len(sentences))]
        if 'qasrl' in self.annotation_layers:
            # verbs detection for qasrl (POS-based) - keep dictionary for all the verbs in the sentence
            predicate_lists = self.verbal_predicate_detector(sentences_tokens_tags, sentences_pos, sentences_lemma)
            # run QA geneation model
            if verbose: print(f"Running QA-SRL QA-generation...")
            outputs_qasrl = self.get_qasrl_qa(sentences, predicate_lists, 'verbal', **generate_kwargs)

        outputs_disc = [[] for k in range(len(sentences))]
        if 'qadiscourse' in self.annotation_layers:
            # QADiscourse model is sentence-level, not grouped by predicates 
            # run qa_discourse pipeline
            if verbose: print(f"Running QADiscourse QA-generation...")
            outputs_disc = self.qa_discourse_pipeline(sentences)

        # Collect outputs of various annotation layers
        outputs = []
        # all `outputs_...` objects are lists corresponding to sentences
        for output_nom, output_qasrl, output_disc in zip(outputs_nom, outputs_qasrl, outputs_disc):
            d = {'qanom': output_nom,
                'qasrl': output_qasrl,
                'qadiscourse': output_disc}
            outputs.append({key: value for key, value in d.items() if key in self.annotation_layers})

        # Open Information Extraction conversion
        if output_openie:
            # convert QA outputs to OpenIE tuples
            qa_outputs = outputs # `outputs` would be a dict including an "openie" section (tuples) beside a "qasem" section (QAs)
            oie_outputs = [self.openie_converter.convert_single_sentence(sent_info, sentence) 
                       for sent_info, sentence in zip(outputs, sentences)]
            outputs = {"qasem": qa_outputs, "openie": oie_outputs}

        return outputs

    def get_qasrl_qa(self, sentences, predicate_lists, predicate_type, **generate_kwargs):
        
        outputs_qa = [[] for k in range(len(sentences))]
        inputs_to_qa_model, input_sentence_index, inputs_verb_forms, inputs_pred_infos = self._prepare_input_sentences(
            sentences, predicate_lists)
        # Run QA-generation pipeline (invoke inside tqdm to show progress bar)
        model_output = list(tqdm(self.qasrl_pipelines[predicate_type](ListDataset(inputs_to_qa_model),
                                        verb_form=inputs_verb_forms,
                                        # verb_form='',
                                        predicate_type=predicate_type, 
                                        **generate_kwargs)))


        if len(inputs_to_qa_model) > 0:
            if self.contextualize:
                # for contextualization
                model_output = self.contextual_qa(model_output, inputs_pred_infos, sentences, input_sentence_index)

        # collect QAs for all predicates in sentence
        for model_pred_output, pred_info, sent_index in zip(model_output, inputs_pred_infos, input_sentence_index):
            predicates_full_info = dict(QAs=model_pred_output['QAs'], **pred_info)
            if NO_REPEAT:
                for i, qa in enumerate(predicates_full_info['QAs']):
                    answers_wo_rep = get_answers_without_repetitions(qa['answers'])
                    predicates_full_info['QAs'][i]['answers'] = answers_wo_rep
            outputs_qa[sent_index].append(predicates_full_info)

        return outputs_qa


    def _prepare_input_sentence(self, raw_sentence: str, predicate_idx: int) -> str:
        words = raw_sentence.split(" ") 
        words = words[:predicate_idx] + ["<predicate>"] + words[predicate_idx:] 
        return " ".join(words)

    def _prepare_input_sentences(self, raw_sentences: Iterable[str], predicate_infos_for_sentences):
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


    def pos_tag_tokens(self, sentences):
        nlp = get_spacy(self.spacy_model_name)
        tokenizer = Tokenizer(nlp.vocab)
        nlp.tokenizer = tokenizer
        spacy_parsed_sentences = list(nlp.pipe(sentences))
        sentences_tokens_tags = []
        sentences_pos = []
        sentences_lemma = []
        for sentence_nlp in spacy_parsed_sentences:
            tag_sent = [(w.text, w.tag_) for w in sentence_nlp]
            pos_sent = [(w.text, w.pos_) for w in sentence_nlp]
            lemma_sent = [w.lemma_ for w in sentence_nlp]
            sentences_tokens_tags.append(tag_sent)
            sentences_pos.append(pos_sent)
            sentences_lemma.append(lemma_sent)
        return sentences_tokens_tags, sentences_pos, sentences_lemma

    def verbal_predicate_detector(self, sentences_tokens_tags, sentences_pos, sentences_lemma):
        predicate_lists = [[]] * len(sentences_tokens_tags)
        for i, sent_tokens_and_poses in enumerate(sentences_pos):
            target_idxs = []
            verb_forms = []
            sent_tokens, sent_pos = zip(*sent_tokens_and_poses)
            lemmas = sentences_lemma[i]
            for j, token in enumerate(sent_pos):
                if token == 'VERB':
                    target_idxs.append(j)
                    verb_forms.append(lemmas[j])
                # elif j != 0 and token == 'ADJ':
                #     k = 1
                #     while sent_pos[j - k] == 'ADV' and j - k > 0: k += 1
                #     if sent_pos[j-k] == 'AUX':
                #         target_idxs.append(j)
                #         verb_forms.append(lemmas[j])


            predicate_lists[i] = [
                {"predicate_idx": pred_idx,
                 "predicate": sent_tokens[pred_idx],
                 "verb_form": verb_form}
                for pred_idx, verb_form in zip(target_idxs, verb_forms)]

        return predicate_lists


    def contextual_qa(self, model_output, pred_infos, sentences, sentences_index):
        context_samples = []
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
        return model_output

# Keep spacy model a global singleton
spacy_models = {}
def get_spacy(lang_model):
    if lang_model not in spacy_models:
        try:
            nlp = spacy.load(lang_model)
        except OSError:
            print(f'Downloading SpaCy model {lang_model} for POS tagging (one-time)...\n', file=sys.stderr)
            spacy.cli.download(lang_model)
            nlp = spacy.load(lang_model)
        spacy_models[lang_model] = nlp
    return spacy_models[lang_model]

def nltk_pos_tag(*inputs):
    """
    In the QASRL/QANom annotation process, they used nltk's pos_tagger ('averaged_perceptron_tagger'),
    So one should use this pos-tagger if concerned with maximum compatability with these datasets.
    Otherwise, SpaCy POS-tagger should work fine.
    """
    tagger_package = 'averaged_perceptron_tagger'
    nltk_downloader = Downloader()
    if not nltk_downloader.is_installed(tagger_package):
        nltk.download(tagger_package)
    return nltk.pos_tag(*inputs)


def get_answers_without_repetitions(answers: List[Tuple[str, int]]):
    answers_no_repetitions = answers.copy()
    i = 1
    while i < len(answers_no_repetitions):
        if answers_no_repetitions[i] in answers_no_repetitions[i - 1]:
            del answers_no_repetitions[i]
        else:
            i += 1
    return answers_no_repetitions

if __name__ == "__main__":
    open_ie_kwargs = {
        "layers_included": ["qasrl", "qanom"],
        "labeled_adjuncts": True,
    }
    pipe = QASemEndToEndPipeline(nominalization_detection_threshold=0.8, openie_converter_kwargs=open_ie_kwargs)
    import sys
    if len(sys.argv)==1:
        sentences = [s.strip() for s in """
        He did not return to military life until the outbreak of the revolution in 1775 .
        Very little further erosion takes place after the formation of a pavement , and the ground becomes stable .
        Moreover , Russia does not want a division of Ukraine , which could lead NATO to become established within the borders of the ex-USSR , so it is more likely it is seeking to change the facts on the ground so to be able to negotiate from a position of strength .
        The king was , at first , put off by her strict religious practice , but he warmed to her through her care for his children .
        An unexpected series of experimental results for the rate of decay of heavy highly charged radioactive ions circulating in a storage ring has provoked theoretical activity in an effort to find a convincing explanation .
        The President said that he views the achievements of the Paralympics above that of the Olympics , that the Paralympics play a vital role in the lives of the athletes .
        """.strip().split("\n")]
    else:
        sentences = sys.argv[1:]
    print(pipe(sentences, output_openie=True))
