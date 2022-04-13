import codecs
import spacy
import random

nlp = spacy.load("en_core_web_sm")

import pandas as pd

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


def create_data_for_question_transformation(question_file, split):
    # ID pred arg : ok
    source_file = codecs.open(split+'.source', 'w')
    target_file = codecs.open(split+'.target', 'w')

    for file_numb, file in enumerate([question_file]):
        input_file = pd.read_csv(file, sep='\t')
        for index, row in input_file.iterrows():
            if isinstance(row['filled_question'], str):
                sentence = row['text'].split()
                pred = row['predicate_lemma']
                proto_question = row['proto_question']
                filled_questions = row['filled_question'].split('~!~')
                filled_question = random.choice(filled_questions)
                predicate_index = int(row['predicate_span'].split(':')[0])
                marked_sentence = []
                for token_idx, token in enumerate(sentence):
                    if token_idx == predicate_index:
                        marked_sentence.append('PREDICATE_START')
                        marked_sentence.append(token)
                        marked_sentence.append('PREDICATE_END')
                    else:
                        marked_sentence.append(token)
                doc = nlp(pred)
                predicate_lemma = ''
                for token in doc:
                    predicate_lemma = token.lemma_
                source_file.write(' '.join(marked_sentence) + ' </s> ' + predicate_lemma + ' [SEP] ' + proto_question +'\n')
                target_file.write(filled_question + '\n')
    source_file.close()
    target_file.close()

if __name__ == "__main__":
    #File path to the train and dev questions: (qasrl_qanom.filled.dev.tsv)
    question_file_dev = ''
    question_file_train = ''
    create_data_for_question_transformation(question_file=question_file_dev, split='val')
    create_data_for_question_transformation(question_file=question_file_train, split='train')