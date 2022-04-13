from bart_transformation.question_translation import QuestionTranslator
from collections import defaultdict
import csv
import jsonlines
import re
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from srl_as_qa_parser import PropBankRoleFinder

def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('/home/nlp/pyatkiv/workspace/CrossSRL/Data/QASRLAlignedQs/qasrl.prototype_accuracy.cardie.tsv') as csvfile:
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
    with open('prototypes/qasrl.prototype_accuracy.adjuncts.tsv') as csvfile:
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

def get_questions(infile_path, outfile_path, pretrained_path):
    modifier_dict = {"PRP": "Secondary Predication", "GOL": "Goal", "DIR": "Directionals", "LOC": "Locatives", "MNR": "Manner", "EXT": "Extent", "REC": "Reciprocals", "PRD": "Secondary Predication", "PNC": "Purpose", "CAU": "Cause", "DIS": "Discourse", "ADV": "adverbials", "MOD": "modals", "NEG": "negation", "TMP": "temporal"}
    role_finder = PropBankRoleFinder.from_framefile('/home/nlp/pyatkiv/workspace/CrossSRL/Data/frames.jsonl')
    q_translator = QuestionTranslator.from_pretrained(pretrained_path, device_id=0)
    proto_dict_adjuncts = get_adjunct_proto_question_dict()
    proto_dict = get_proto_question_dict()
    fieldnames = ['doc_id', 'sent_id', 'questions', 'roles', "predicate_span", 'text', "role_descriptions", "predicate_lemma", "experiment_type"]
    outfile = csv.DictWriter(open(outfile_path, 'w'), fieldnames=fieldnames)
    outfile.writeheader()
    infile = csv.DictReader(open(infile_path, 'r'), delimiter='\t')
    pred_lemma_set = []
    for s_n, sentence_entry in enumerate(infile):
        role_descriptions = []
        doc_id = str(s_n)
        sent_id = str(s_n)
        text = sentence_entry["text"]
        predicate_lemma = sentence_entry["predicate_sense"].split('.')[0]
        pred_lemma_set.append(predicate_lemma)
        predicate_sense = sentence_entry["predicate_sense"].split('.')[1]
        role = sentence_entry["pb_role_type"]
        predicate_span = sentence_entry["predicate_span"]
        pos = sentence_entry["pos"]
        the_roles = role_finder.get_roles(predicate_lemma, pos=pos, sense=int(predicate_sense))
        for role_tuple in the_roles:
            role_description = role_tuple[1]
            role_descriptions.append(role_description)
        role_descriptions.append('Location')
        role_descriptions.append('Manner')
        if role.startswith('AM'):
            proto_question = proto_dict_adjuncts[role]
            proto_question = re.sub('<PLACEHOLDER>', predicate_lemma, proto_question)

        else:
            if predicate_lemma == 'trial':
                predicate_lemma='try'
            proto_question = proto_dict[predicate_lemma + predicate_sense + role]
        if proto_question == '':
            verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
            if found:
                for verb in verbs:
                    if verb + predicate_sense + role in proto_dict:
                        proto_question = proto_dict[verb + predicate_sense + role]
        if proto_question == '':
            pass
        samples = []
        samples.append({'proto_question': proto_question, 'predicate_lemma': predicate_lemma, 'predicate_span': predicate_span, 'text': text})
        #TODO
        questions = []
        #questions = q_translator.predict(samples)
        questions.append(sentence_entry["query_question"])
        questions = '~!~'.join(questions)
        #roles = role+'~!~'+role+'-C'
        roles = role
        role_descriptions = '~!~'.join(role_descriptions)
        outfile.writerow({"text": text, "doc_id": doc_id, "sent_id": sent_id, "questions": questions, "roles": roles, "predicate_span": predicate_span, "role_descriptions": role_descriptions, "predicate_lemma": predicate_lemma, "experiment_type": "cardie"})
    print(set(pred_lemma_set))

infile_path = '99_ACE2.tsv'
outfile_path = '99_ACE_for_MTurk_only_cardie2.csv'
pretrained_path = '/home/nlp/pyatkiv/workspace/transformers/examples/seq2seq/question_transformation_grammar_corrected_who/'
get_questions(infile_path, outfile_path, pretrained_path)