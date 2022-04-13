# from typing import Iterable, Optional

# # from qanom.nominalization_detector import NominalizationDetector
# from nominalization_detector import NominalizationDetector
# # from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline
# from qasrl_seq2seq_pipeline import QASRL_Pipeline
import spacy

qasrl_context_model = "/home/nlp/wolhanr/QASem/RoleQGeneration/question_transformation_grammar_corrected_who"

# default_model = "baseline"


# class QASrlContextEndToEndPipeline():
#     """
#     This pipeline wraps QAnom pipeline and qasrl pipeline
#     """
#     def __init__(self, 
#                  qasrl_model: Optional[str] = None, 
#                 ):
        
#         qanom_model = qanom_model or default_model
#         model_url = qasrl_context_model 

#     def __call__(self, sentences: Iterable[str], 
#                  **generate_kwargs):

#         return 1
        



# if __name__ == "__main__":
#     pipe = QASrlContextEndToEndPipeline()
#     # sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
#     # print(pipe([sentence])) 
#     #res1 = pipe(["The student was interested in Luke 's research about see animals ."])#, verb_form="research", predicate_type="nominal")
#     res2 = pipe(["The doctor was interested in Luke 's treatment .", "The Veterinary student was interested in Luke 's treatment of sea animals .", "I eat a peach"])#, verb_form="treat", predicate_type="nominal", num_beams=10)
#     #res3 = pipe(["A number of professions have developed that specialize in the treatment of mental disorders ."])
#     # print(res1)
#     print(res2)
#     # print(res3) 
#     # 
#     # 




import csv
import re
from collections import defaultdict
from candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from RoleQGeneration.srl_as_qa_parser import PropBankRoleFinder
from RoleQGeneration.question_translation import QuestionTranslator
from argparse import ArgumentParser


def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('/home/nlp/wolhanr/QASem/RoleQGeneration/resources/qasrl.prototype_accuracy.ontonotes.tsv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sense_id = row["sense_id"]
            if row['verb_form']+sense_id+row['role_type'] in proto_dict:
                current_count = proto_score[row['verb_form']+sense_id +row['role_type']]
                considered_count = float(row['squad_f1'])
                if considered_count>current_count:
                    proto_dict[row['verb_form'] +sense_id +row['role_type']] = row['proto_question']
                    proto_score[row['verb_form'] +sense_id +row['role_type']] = considered_count
            else:
                proto_dict[row['verb_form']+sense_id +row['role_type']]=row['proto_question']
                proto_score[row['verb_form'] + sense_id + row['role_type']] = float(row['squad_f1'])
    return proto_dict


def get_adjunct_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('/home/nlp/wolhanr/QASem/RoleQGeneration/resources/qasrl.prototype_accuracy.adjuncts.tsv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row['role_type'] in proto_dict:
                current_count = proto_score[row['role_type']]
                considered_count = float(row['squad_f1'])
                if considered_count>current_count:
                    proto_dict[row['role_type']] = row['proto_question']
                    proto_score[row['role_type']] = considered_count
            else:
                proto_dict[row['role_type']]=row['proto_question']
                proto_score[row['role_type']] = float(row['squad_f1'])
    return proto_dict


def get_adjuncts(q_translator, predicate_lemma, predicate_span, text):
    adjunct_dict = {}
    proto_dict_adjuncts = get_adjunct_proto_question_dict()
    roles = ['AM-LOC', 'AM-MNR', 'AM-CAU', 'AM-EXT', 'AM-GOL']
    role_descriptions = ['Location', 'Manner', 'Causal', 'Extent', 'Goal']
    samples = []
    for role in roles:
        proto_question = proto_dict_adjuncts[role]
        proto_question = re.sub('<PLACEHOLDER>', predicate_lemma, proto_question)
        if proto_question == '':
            pass
        samples.append(
            {'proto_question': proto_question, 'predicate_lemma': predicate_lemma, 'predicate_span': predicate_span,
             'text': text})
    contextualized_questions = q_translator.predict(samples)
    for question, role, role_description in zip(contextualized_questions, roles, role_descriptions):
        adjunct_dict[role+'_'+role_description]=question
    return adjunct_dict


def get_questions(sentences, transformation_model_path, device_number, with_adjuncts=False):
    role_finder = PropBankRoleFinder.from_framefile('/home/nlp/wolhanr/QASem/RoleQGeneration/role_lexicon/frames.jsonl')
    #Generating Question Transformation
    q_translator = QuestionTranslator.from_pretrained(transformation_model_path, device_id=int(device_number))

    proto_dict = get_proto_question_dict()
    # outfile = jsonlines.open(outfile, mode='w')
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
        {"target_idx": pred_idx,
            "predicate": tokens_text[pred_idx],
            "target_lemma": verb_form} 
        for pred_idx, verb_form in zip(target_idxs, verb_forms)]

    outputs_qasrl = []
    for sentence, predicate_infos in zip(sentences, predicate_lists):
        # collect QAs for all predicates in sentence 
        predicates_full_infos = []
        for pred_info in predicate_infos:
            text = sentence
            pos = "v"
            predicate_index = pred_info["target_idx"]
            predicate_span = str(pred_info["target_idx"])+':'+str(pred_info["target_idx"]+1)
            predicate_lemma = pred_info["target_lemma"]
            predicate_sense = "1" # default 1
            # get roles and role descriptions
            the_roles = role_finder.get_roles(predicate_lemma, pos=pos,sense=int(predicate_sense))
            questions = {}
            samples = []
            roles = []
            role_descriptions = []
            for role_tuple in the_roles:
                role = role_tuple[0]
                roles.append(role)
                role_description = role_tuple[1]
                role_descriptions.append(role_description)
                # get proto questions
                proto_question = proto_dict[predicate_lemma+predicate_sense+role]
                # If you need to work with nominal inputs:
                if proto_question == '':
                    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
                    if found:
                        for verb in verbs:
                            if verb+predicate_sense+role in proto_dict:
                                proto_question = proto_dict[verb + predicate_sense + role]
                if proto_question=='':
                    questions[role+'_'+role_description] = "No Prototype"
                else:
                    samples.append(
                        {'proto_question': proto_question, 'predicate_lemma': predicate_lemma,
                        'predicate_span': predicate_span,
                        'text': text})
            if samples == []:
                    predicates_full_info = {"sentence": text, "target_idx": predicate_index, "target_lemma": predicate_lemma,
                    "target_pos": pos, "predicate_sense": predicate_sense, "questions": "PREDICATE IS NOT IN ROLE ONTOLOGY",
                    "adjunct_questions": "PREDICATE IS NOT IN ROLE ONTOLOGY"}
            else:
                # contextualize the questions
                contextualized_questions = q_translator.predict(samples)
                for question, role, role_description in zip(contextualized_questions, roles, role_descriptions):
                    questions[role+'_'+role_description] = question
                adjunct_question_dict = {}
                if with_adjuncts:
                    adjunct_question_dict = get_adjuncts(q_translator, predicate_lemma, predicate_span, text)
                predicates_full_info = {"sentence": text, "target_idx": predicate_index, "target_lemma": predicate_lemma, "target_pos": pos, "predicate_sense": predicate_sense, "questions": questions, "adjunct_questions":adjunct_question_dict}
            predicates_full_infos.append(predicates_full_info)
        outputs_qasrl.append(predicates_full_infos)

    return outputs_qasrl

a = get_questions(sentences=["Tom brings the dog to the park."], transformation_model_path = qasrl_context_model , device_number=0, with_adjuncts=True)
print(a)


