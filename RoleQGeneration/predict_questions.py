import jsonlines
import csv
import re
from collections import defaultdict
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from srl_as_qa_parser import PropBankRoleFinder
from question_translation import QuestionTranslator
from argparse import ArgumentParser


def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('resources/qasrl.prototype_accuracy.ontonotes.tsv') as csvfile:
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
    with open('resources/qasrl.prototype_accuracy.adjuncts.tsv') as csvfile:
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


def get_questions(infile, outfile, transformation_model_path, device_number, with_adjuncts):
    role_finder = PropBankRoleFinder.from_framefile('role_lexicon/frames.jsonl')
    #Generating Question Transformation
    q_translator = QuestionTranslator.from_pretrained(transformation_model_path, device_id=int(device_number))

    proto_dict = get_proto_question_dict()
    outfile = jsonlines.open(outfile, mode='w')

    infile = jsonlines.open(infile)
    for row in infile:
        instance_id = row["id"]
        text = row["sentence"]
        pos = row["target_pos"]
        predicate_index = row["target_idx"]
        predicate_span = str(row["target_idx"])+':'+str(row["target_idx"]+1)
        predicate_lemma = row["target_lemma"]
        predicate_sense = str(row["predicate_sense"])
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
            outfile.write(
                {"id": instance_id, "sentence": text, "target_idx": predicate_index, "target_lemma": predicate_lemma,
                 "target_pos": pos, "predicate_sense": predicate_sense, "questions": "PREDICATE IS NOT IN ROLE ONTOLOGY",
                 "adjunct_questions": "PREDICATE IS NOT IN ROLE ONTOLOGY"})
        else:
            #contextualize the questions
            contextualized_questions = q_translator.predict(samples)
            for question, role, role_description in zip(contextualized_questions, roles, role_descriptions):
                questions[role+'_'+role_description] = question
            adjunct_question_dict = {}
            if with_adjuncts:
                adjunct_question_dict = get_adjuncts(q_translator, predicate_lemma, predicate_span, text)
            outfile.write({"id": instance_id, "sentence": text, "target_idx": predicate_index, "target_lemma": predicate_lemma, "target_pos": pos, "predicate_sense": predicate_sense, "questions": questions, "adjunct_questions":adjunct_question_dict})


def main(args):
    get_questions(args.infile, args.outfile, args.transformation_model_path, args.device_number, args.with_adjuncts)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--infile", help="debug_file.jsonl")
    ap.add_argument("--outfile", help="name of the file you want to write the question to (jsonl format)")
    ap.add_argument("--transformation_model_path")
    ap.add_argument("--device_number", default="0")
    ap.add_argument("--with_adjuncts", default=False)

    main(ap.parse_args())

