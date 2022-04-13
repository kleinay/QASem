import pandas as pd
import jsonlines
from tqdm import tqdm

import lemma_utils
from qa_models import parse_span


def read_ontonotes_df(onto_jsonl_path):
    data = []
    sentences = list(jsonlines.open(onto_jsonl_path))
    for sent in tqdm(sentences):
        tokens = sent['text'].split()
        for frame in sent['frames']:
            predicate = frame['predicate']
            if 'predicate_lemma' not in predicate and 'lemma' not in predicate:
                pred_idx, _ = parse_span(predicate['span'])
                lemma = lemma_utils.get_lemma(tokens, pred_idx)
                predicate['lemma'] = lemma
            if 'lemma' not in predicate:
                predicate['lemma'] = predicate['predicate_lemma']
                del predicate['predicate_lemma']

            if 'verb_form' not in predicate:
                predicate['verb_form'] = lemma_utils.get_verb_form(predicate['lemma'])

            for arg in frame['arguments']:
                d = {
                    "predicate_lemma": predicate['lemma'],
                    "predicate_pos": predicate['pos'],
                    "verb_form": predicate['verb_form'],
                    "predicate_span": predicate['span'],
                    "sense_id": predicate['frame_id'],
                    "argument": arg['text'],
                    "argument_span": arg['span'],
                    "role_type": arg['role_type'],
                    'arg_type': arg['arg_type'],
                    "argument_sent_id": arg['sent_id'],
                    "sent_id": sent['sent_id'],
                    "head_idx": arg['head_idx'],
                    "doc_id": sent['doc_id'],
                    "text": sent['text']
                }
                data.append(d)
    df = pd.DataFrame(data)
    # with jsonlines.open(onto_jsonl_path, "w") as fout:
    #     fout.write_all(sentences)
    return df
from sklearn.metrics import classification_report
