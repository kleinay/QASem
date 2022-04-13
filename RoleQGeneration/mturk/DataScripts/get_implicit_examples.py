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

def lemma_sense_to_roles(frame_path):
    frame_file = csv.DictReader(open(frame_path), delimiter='\t')
    lemma_sense_role_dict = defaultdict(lambda : [])
    for row in frame_file:
        lemma_sense_role_dict[row["predicate_lemma"]+' '+row["sense_id"]].append(row["role_type"])
    return lemma_sense_role_dict

def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('/home/nlp/pyatkiv/workspace/CrossSRL/Data/QASRLAlignedQs/qasrl.prototype_accuracy.gc_moor.tsv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sense_id = str(int(row["sense_id"]))
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

def create_csv(implicit_file, outfile, transformation_model_path):
    default_role_dict = {"A0": "Agent", "A1": "Patient", "A2": "Instrument, Benefactive, Attribute", "A3": "Starting Point, Benefactive, Attribute", "A4": "Destination, Ending Point"}
    modifier_dict = {"PRP": "Secondary Predication", "GOL": "Goal", "DIR": "Directionals", "LOC": "Locatives", "MNR": "Manner", "EXT": "Extent", "REC": "Reciprocals", "PRD": "Secondary Predication", "PNC": "Purpose", "CAU": "Cause", "DIS": "Discourse", "ADV": "adverbials", "MOD": "modals", "NEG": "negation", "TMP": "temporal"}

    #Generating Question Transformation
    transformation_model = BartForConditionalGeneration.from_pretrained(transformation_model_path)
    device = torch.device(1)
    transformation_model.to(device)
    transformation_tokenizer = BartTokenizer.from_pretrained(transformation_model_path)

    proto_dict = get_proto_question_dict()
    desc_dict = load_frames('/home/nlp/pyatkiv/workspace/CrossSRL/Data/predicate_roles.gc_moor.tsv')
    lemma_sense_roles_dict = lemma_sense_to_roles('/home/nlp/pyatkiv/workspace/CrossSRL/Data/predicate_roles.gc_moor.tsv')
    fieldnames = ['doc_id', 'sent_id', 'questions', 'roles', "predicate_span", 'text', "role_descriptions", "predicate_lemma", "predicate_sense"]
    outfile = csv.DictWriter(open(outfile, 'w'), fieldnames=fieldnames)
    outfile.writeheader()
    infile = jsonlines.open(implicit_file, 'r')
    for sentence_entry in infile:
        text = sentence_entry["text"]
        doc_id = sentence_entry["doc_id"]
        predicate_frame = sentence_entry["predicate"]
        predicate_span = predicate_frame["span"]
        predicate_index = int(predicate_span.split(':')[0])
        predicate_lemma = predicate_frame["predicate_lemma"]
        sent_id = sentence_entry["sent_id"]
        try:
            predicate_sense = str(int(predicate_frame["frame_id"]))
        except KeyError:
            predicate_sense = '1'
        arguments = sentence_entry["arguments"]
        questions = []
        roles = []
        arg_spans = []
        answer_options = []
        role_descriptions = []
        found_implicit = False
        explicit_roles = []
        for arg in arguments:
            arg_span = arg["span"]
            role = arg["role_type"]
            explicit_roles.append(role)
            arg_type = arg["arg_type"]
            position = arg["position"]
            if arg_type == 'implicit':
                found_implicit = True
            arg_spans.append(arg_span)
            answer_options.append(arg_span)
        if found_implicit:
            the_roles = lemma_sense_roles_dict[predicate_lemma+' '+predicate_sense]
            for role in the_roles:
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
                role_descriptions.append(role_description)
                proto_question = proto_dict[predicate_lemma+predicate_sense+role]
                if proto_question == '':
                    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
                    if found:
                        for verb in verbs:
                            if verb + predicate_sense + role in proto_dict:
                                proto_question = proto_dict[verb + predicate_sense + role]
                if proto_question=='':
                    print(predicate_lemma+predicate_sense+role)
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
                outfile.writerow({"text": text, "doc_id": doc_id, "sent_id": sent_id, "questions": questions, "roles": roles, "predicate_span": predicate_span, "role_descriptions": role_descriptions, "predicate_lemma": predicate_lemma, "predicate_sense": predicate_sense})


implicit_file = '/home/nlp/pyatkiv/workspace/CrossSRL/Data/GC/gerber_chai_test.jsonl'
outfile = 'gc.test_only_implicit.csv'
transformation_model_path = '/home/nlp/pyatkiv/workspace/transformers/examples/seq2seq/question_transformation_grammar_corrected_who/'
create_csv(implicit_file, outfile, transformation_model_path)