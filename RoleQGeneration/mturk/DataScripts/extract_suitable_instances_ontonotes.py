import jsonlines
import random
import csv
import torch
from collections import defaultdict
from transformers import BartTokenizer, BartForConditionalGeneration
import os
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from argparse import ArgumentParser

if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from srl_as_qa_parser import PropBankRoleFinder


def get_onto_dicts(ontonotesfile):
    #doc_id + sent_id : entry
    onto_dict = defaultdict(lambda : '')
    #predicate_lemma + predicate_sense + role : [doc_id + sent_id]
    role_onto_dict = defaultdict(lambda : [])
    #predicate_lemma : predicate_sense : [doc_id + sent_id]
    pred_sense_dict = defaultdict(lambda : defaultdict(lambda : []))
    infile = jsonlines.open(ontonotesfile, 'r')
    for sentence_entry in infile:
        text = sentence_entry["text"]
        if len(text.split())>=5:
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
                        if role[0] not in ['R', 'C'] and not role.startswith('AM') and not role.startswith('AA'):
                            role_onto_dict[role].append(doc_id+' '+sent_id+ ' ' + predicate_idx)
    return onto_dict, role_onto_dict, pred_sense_dict

def get_random_instances(covered, infile, outfile):
    infile = csv.DictReader(open(infile))
    header = infile.fieldnames
    outfile = csv.DictWriter(open(outfile, 'w'), fieldnames=header)
    outfile.writeheader()
    candidates = []
    for entry in infile:
        text = entry["text"]
        if len(text.split())>=5:
            doc_id = entry["doc_id"]
            sent_id = entry["sent_id"]
            predicate_idx = entry["predicate_span"]
            if doc_id+' '+sent_id+' '+predicate_idx not in covered:
                candidates.append(entry)
    sampled_candidates = random.sample(candidates, 300)
    for candidate in sampled_candidates:
        outfile.writerow(candidate)



def get_stratified_sample(outfile, amount, transformation_model_path, ontonotesfile):
    covered = []
    onto_dict, role_onto_dict, pred_sense_dict = get_onto_dicts(ontonotesfile)
    sense_count_dict = defaultdict(lambda : [])
    for predicate, sense_dict in pred_sense_dict.items():
        sense_count_dict[len(sense_dict.keys())].append(predicate)
    instances = []
    for i in range(1, 6):
        predicates = sense_count_dict[i]
        sampled_predicate = random.sample(predicates, 1)[0]
        sense_ids = pred_sense_dict[sampled_predicate].keys()
        for sense_id in sense_ids:
            pred_s_instances = pred_sense_dict[sampled_predicate][str(sense_id)]
            pred_s_instance = random.sample(pred_s_instances, 1)[0]
            instances.append(onto_dict[pred_s_instance])
            covered.append(pred_s_instance)
    for role, entries in role_onto_dict.items():
        candidates = []
        for entry in entries:
            if entry not in covered:
                candidates.append(entry)
        amount = 50
        if len(candidates) < 50:
            amount = len(candidates)
        role_samples = random.sample(candidates, amount)
        for sample in role_samples:
            if sample not in covered:
                instances.append(onto_dict[sample])
                covered.append(sample)
    get_random_instances(covered, amount)
    get_questions(instances, outfile, transformation_model_path)

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

def get_questions(instances, outfile, transformation_model_path, frame_file):

    role_finder = PropBankRoleFinder.from_framefile(frame_file)
    #default_role_dict = {"A0": "Agent", "A1": "Patient", "A2": "Instrument, Benefactive, Attribute", "A3": "Starting Point, Benefactive, Attribute", "A4": "Destination, Ending Point"}
    #modifier_dict = {"PRP": "Secondary Predication", "GOL": "Goal", "DIR": "Directionals", "LOC": "Locatives", "MNR": "Manner", "EXT": "Extent", "REC": "Reciprocals", "PRD": "Secondary Predication", "PNC": "Purpose", "CAU": "Cause", "DIS": "Discourse", "ADV": "adverbials", "MOD": "modals", "NEG": "negation", "TMP": "temporal"}

    #Generating Question Transformation
    transformation_model = BartForConditionalGeneration.from_pretrained(transformation_model_path)
    device = torch.device(3)
    transformation_model.to(device)
    transformation_tokenizer = BartTokenizer.from_pretrained(transformation_model_path)

    proto_dict = get_proto_question_dict()
    fieldnames = ['doc_id', 'sent_id', 'questions', 'roles', "predicate_span", 'text', "role_descriptions"]
    outfile = csv.DictWriter(open(outfile, 'w'), fieldnames=fieldnames)
    outfile.writeheader()

    for sentence_entry in instances:
        text = sentence_entry["text"]
        doc_id = sentence_entry["doc_id"]
        for frame in sentence_entry["frames"]:
            predicate_frame = frame["predicate"]
            predicate_span = predicate_frame["span"]
            predicate_index = int(predicate_span.split(':')[0])
            predicate_lemma = predicate_frame["predicate_lemma"]
            sent_id = predicate_frame["sent_id"]
            predicate_sense = str(int(predicate_frame["frame_id"]))
            arguments = frame["arguments"]
            questions = []
            roles = []
            arg_spans = []
            answer_options = []
            role_descriptions = []
            for arg in arguments:
                arg_span = arg["span"]
                role = arg["role_type"]
                arg_type = arg["arg_type"]
                position = arg["position"]
                arg_spans.append(arg_span)
                answer_options.append(arg_span)
                if role[0] in ['R', 'C'] or arg_type == 'implicit' or position=='cross_sent':
                    pass
                else:
                    arg_spans.append(arg_span)
                    answer_options.append(arg_span)
            pos = predicate_frame['pos'][0].lower()
            the_roles = role_finder.get_roles(predicate_lemma, pos=pos,sense=int(predicate_sense))
            for role_tuple in the_roles:
                role = role_tuple[0]
                role_description = role_tuple[1]
                role_descriptions.append(role_description)
                proto_question = proto_dict[predicate_lemma+predicate_sense+role]
                if proto_question == '':
                    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
                    if found:
                        for verb in verbs:
                            if verb+predicate_sense+role in proto_dict:
                                proto_question = proto_dict[verb + predicate_sense + role]
                if proto_question=='':
                    pass
                else:
                    marked_sentence = []
                    for token_idx, token in enumerate(text.split()):
                        if token_idx == predicate_index:
                            marked_sentence.append('PREDICATE_START')
                            marked_sentence.append(token)
                            marked_sentence.append('PREDICATE_END')
                        else:
                            marked_sentence.append(token)
                    input_text = ' '.join(marked_sentence)
                    ARTICLE_TO_SUMMARIZE = input_text + ' </s> ' + predicate_lemma + ' [SEP] ' + proto_question
                    inputs = transformation_tokenizer([ARTICLE_TO_SUMMARIZE], max_length=500, return_tensors='pt')
                    inputs = inputs.to(device)
                    # Generate Summary
                    summary_ids = transformation_model.generate(inputs['input_ids'], num_beams=1, max_length=30, early_stopping=True)
                    summary_ids = summary_ids.detach().cpu()
                    prediction = [transformation_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids][0]
                    contextualized_question = prediction.split('?')[0] + '?'
                    questions.append(contextualized_question)
                    roles.append(role)
            if len(questions)>0:
                questions = '~!~'.join(questions)
                roles = '~!~'.join(roles)
                role_descriptions = '~!~'.join(role_descriptions)
                outfile.writerow({"text": text, "doc_id": doc_id, "sent_id": sent_id, "questions": questions, "roles": roles, "predicate_span": predicate_span, "role_descriptions": role_descriptions})

def main(args):
    get_stratified_sample(args.outfile, args.amount, args.transformation_model_path, args.ontonotesfile)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-outfile", help="/path/to/outputfile.csv")
    ap.add_argument("-amount", help="number of instances to extract")
    ap.add_argument("-transformation_model_path", help="/path/to/trained_BART_model")
    ap.add_argument("-ontonotesfile", help="/path/to/ontonotes")

    main(ap.parse_args())
