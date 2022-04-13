import jsonlines
import csv
import torch
from collections import defaultdict
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import os
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources


if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from srl_as_qa_parser import PropBankRoleFinder
# from ontonotes dev (and maybe test)
# ID, text, Q1_Q2, A1-A2
# doc_id, sent_id, questions, roles, predicate_span, text
# predicate with many verb senses

# tip, bring, submit, speak, shut, save, return, sail, lure, lose

def load_frames(frame_path):
    frames = pd.read_csv(frame_path, sep="\t")
    frames = frames[frames.sense_particle == "_"].copy()
    frames = frames[['predicate_lemma', 'role_type', 'role_desc']].drop_duplicates()
    frames = frames.groupby(['predicate_lemma', 'role_type']).head(5)
    frames = frames.groupby(['predicate_lemma', 'role_type']).role_desc.apply(" ; ".join).reset_index()
    frames.sort_values(['predicate_lemma', 'role_type'], inplace=True)
    frames = frames.to_dict(orient='records')
    frame_to_desc = {(f['predicate_lemma'], f['role_type']): f['role_desc']
                     for f in frames}
    return frame_to_desc

def get_onto_dicts():
    #doc_id + sent_id : entry
    onto_dict = defaultdict(lambda : '')
    #predicate_lemma + predicate_sense + role : [doc_id + sent_id]
    role_onto_dict = defaultdict(lambda : [])
    #predicate_lemma : predicate_sense : [doc_id + sent_id]
    pred_sense_dict = defaultdict(lambda : defaultdict(lambda : []))
    ontonotesfile = '/home/nlp/pyatkiv/workspace/CrossSRL/Data/ontonotes/ontonotes.dev.jsonl'
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
                    # TODO: change to include the adjuncts that we want
                    if role[0] not in ['R', 'C'] and not role.startswith('AM') and not role.startswith('AA'):
                        role_onto_dict[role].append(doc_id+' '+sent_id+ ' ' + predicate_idx)
    return onto_dict, role_onto_dict, pred_sense_dict

def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('/home/nlp/pyatkiv/workspace/CrossSRL/Data/QASRLAlignedQs/qasrl.prototype_accuracy_all.tsv') as csvfile:
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

def create_csv(infile_path, outfile, transformation_model_path):
    default_role_dict = {"A0": "Agent", "A1": "Patient", "A2": "Instrument, Benefactive, Attribute", "A3": "Starting Point, Benefactive, Attribute", "A4": "Destination, Ending Point"}
    modifier_dict = {"PRP": "Secondary Predication", "GOL": "Goal", "DIR": "Directionals", "LOC": "Locatives", "MNR": "Manner", "EXT": "Extent", "REC": "Reciprocals", "PRD": "Secondary Predication", "PNC": "Purpose", "CAU": "Cause", "DIS": "Discourse", "ADV": "adverbials", "MOD": "modals", "NEG": "negation", "TMP": "temporal"}
    desc_dict = load_frames('/home/nlp/pyatkiv/workspace/CrossSRL/Data/predicate_roles.tsv')

    role_finder = PropBankRoleFinder.from_framefile('/home/nlp/pyatkiv/workspace/CrossSRL/Data/frames.jsonl')
    onto_dict, role_onto_dict, pred_sense_dict = get_onto_dicts()
    #Generating Question Transformation
    transformation_model = BartForConditionalGeneration.from_pretrained(transformation_model_path)
    device = torch.device(0)
    transformation_model.to(device)
    transformation_tokenizer = BartTokenizer.from_pretrained(transformation_model_path)

    proto_dict = get_proto_question_dict()
    fieldnames = ['doc_id', 'sent_id', 'questions', 'roles', "predicate_span", 'text', "role_descriptions"]
    outfile = csv.DictWriter(open(outfile, 'w'), fieldnames=fieldnames)
    outfile.writeheader()
    infile = csv.DictReader(open(infile_path), delimiter=',')
    for row in infile:
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
            the_roles = role_finder.get_roles(predicate_lemma, pos=pos, sense=int(predicate_sense))
            for role_tuple in the_roles:
                role = role_tuple[0]
                role_description = role_tuple[1]
                role_descriptions.append(role_description)
                role_description = 'no description'
                if (predicate_lemma, role) in desc_dict:
                    role_description = desc_dict[(predicate_lemma, role)]
                else:
                    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
                    if found:
                        for verb in verbs:
                            if (verb, role) in desc_dict:
                                role_description = desc_dict[(verb, role)]
                                break
                if role_description == 'no description' and len(role.split('-')) > 1:
                    role_parts = role.split('-')
                    if role_parts[1] in modifier_dict:
                        role_description = modifier_dict[role_parts[1]]
                    elif role in default_role_dict:
                        role_description = default_role_dict[role]
                proto_question = proto_dict[predicate_lemma+predicate_sense+role]
                if proto_question == '':
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
                    ARTICLE_TO_SUMMARIZE = input_text + ' </s> ' + role_description + ' ' + role + ' ' + predicate_lemma
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


infile = 'Analysis/mturk/random_sample.100.more.csv'
outfile = 'Analysis/mturk/random_sample.100.more.baseline.csv'
transformation_model_path = '/home/nlp/pyatkiv/workspace/transformers/examples/seq2seq/baseline_qasrl/'
create_csv(infile, outfile, transformation_model_path)