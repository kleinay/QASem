from bart_transformation.question_translation import QuestionTranslator
from collections import defaultdict
import csv
import jsonlines
import re
from argparse import ArgumentParser


def get_onto_dicts(ontonotesfile):
    #doc_id + sent_id : entry
    onto_dict = defaultdict(lambda : '')
    #predicate_lemma + predicate_sense + role : [doc_id + sent_id]
    role_onto_dict = defaultdict(lambda : [])
    #predicate_lemma : predicate_sense : [doc_id + sent_id]
    pred_sense_dict = defaultdict(lambda : defaultdict(lambda : []))
    infile = jsonlines.open(ontonotesfile, 'r')
    for sentence_entry in infile:
        doc_id = sentence_entry["doc_id"]
        sent_id = str(sentence_entry["sent_id"])
        for frame in sentence_entry["frames"]:
            predicate_frame = frame["predicate"]
            arguments = frame["arguments"]
            predicate_lemma = predicate_frame["predicate_lemma"]
            predicate_idx = predicate_frame["span"]
            out_entry = sentence_entry
            out_entry["frames"] = [frame]
            onto_dict[doc_id + ' ' + sent_id + ' ' + predicate_idx] = out_entry
            predicate_pos = predicate_frame["pos"]
            if predicate_pos[0] in ['V', 'N']:
                predicate_sense = str(int(predicate_frame["frame_id"]))
                pred_sense_dict[predicate_lemma][predicate_sense].append(doc_id + ' ' + sent_id+ ' ' + predicate_idx)
                for arg in arguments:
                    role = arg["role_type"]
                    role_onto_dict[role].append(doc_id+' '+sent_id+ ' ' + predicate_idx)
    return onto_dict, role_onto_dict, pred_sense_dict


def get_proto_question_dict(proto_file):
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open(proto_file) as csvfile:
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

def get_adjunct_proto_question_dict(adjunct_proto_file):
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open(adjunct_proto_file) as csvfile:
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

def get_questions(infile_path, outfile_path, pretrained_path, ontonotesfile, adjunct_proto_file):
    q_translator = QuestionTranslator.from_pretrained(pretrained_path, device_id=0)
    proto_dict_adjuncts = get_adjunct_proto_question_dict(adjunct_proto_file)
    onto_dict, role_onto_dict, pred_sense_dict = get_onto_dicts(ontonotesfile)
    fieldnames = ['doc_id', 'sent_id', 'questions', 'roles', "predicate_span", 'text', "role_descriptions", "predicate_lemma", "experiment_type"]
    outfile = csv.DictWriter(open(outfile_path, 'w'), fieldnames=fieldnames)
    outfile.writeheader()
    infile = csv.DictReader(open(infile_path, 'r'), delimiter=',')
    for s_n, row in enumerate(infile):
        row = dict(row)
        doc_id = row['doc_id']
        sent_id = row['sent_id']
        predicate_idx = row['predicate_span']
        sentence_entry = onto_dict[doc_id + ' ' + sent_id + ' ' + predicate_idx]
        text = sentence_entry["text"]
        doc_id = sentence_entry["doc_id"]
        for frame in sentence_entry["frames"]:
            predicate_frame = frame["predicate"]
            predicate_span = predicate_frame["span"]
            predicate_lemma = predicate_frame["predicate_lemma"]
            sent_id = predicate_frame["sent_id"]
            role_descriptions = []
            role_descriptions.append('Location')
            role_descriptions.append('Manner')
            role_descriptions.append('Causal')
            role_descriptions.append('Extent')
            role_descriptions.append('Goal')
            roles = ['AM-LOC', 'AM-MNR', 'AM-CAU', 'AM-EXT', 'AM-GOL']
            role_descriptions = '~!~'.join(role_descriptions)
            questions = []
            for role in roles:
                proto_question = proto_dict_adjuncts[role]
                proto_question = re.sub('<PLACEHOLDER>', predicate_lemma, proto_question)
                if proto_question == '':
                    pass
                samples = []
                samples.append({'proto_question': proto_question, 'predicate_lemma': predicate_lemma, 'predicate_span': predicate_span, 'text': text})
                questions.append(q_translator.predict(samples)[0])
            questions = '~!~'.join(questions)
            roles = '~!~'.join(roles)
            outfile.writerow({"text": text, "doc_id": doc_id, "sent_id": sent_id, "questions": questions, "roles": roles, "predicate_span": predicate_span, "role_descriptions": role_descriptions, "predicate_lemma": predicate_lemma, "experiment_type": "onto_adjuncts"})

def main(args):
    infile_path = args.infile
    outfile_path = args.outfile
    pretrained_path = args.adjunct_proto_file
    ontonotesfile = args.ontonotesfile
    adjunct_proto_file = args.adjunct_proto_file
    get_questions(infile_path, outfile_path, pretrained_path, ontonotesfile, adjunct_proto_file)

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-outfile", help="/path/to/outputfile.csv")
    ap.add_argument("-infile")
    ap.add_argument("-adjunct_proto_file")
    ap.add_argument("-transformation_model_path", help="/path/to/trained_BART_model")
    ap.add_argument("-ontonotesfile", help="/path/to/ontonotes")

    main(ap.parse_args())
