
import pandas as pd

import spacy
from jsonlines import jsonlines

from qa_models import parse_span
from qanom.evaluation.metrics import iou

nlp = spacy.load('en_core_web_sm')

# let's find in OntoNotes these sentences and compare argument to intended answer.
ONTONOTES_DEV_PATH = "../../../CrossSRL/Data/ontonotes/ontonotes.dev.jsonl"


def load_ontonotes(onto_path):
    onto_sents = list(jsonlines.open(onto_path))
    onto_data = []
    for sent in onto_sents:
        for frame in sent['frames']:
            predicate = frame['predicate']
            for arg in frame['arguments']:
                if arg['sent_id'] != sent['sent_id']:
                    continue
                onto_data.append({
                    "doc_id": sent['doc_id'],
                    "sent_id": sent['sent_id'],
                    "sense_id": predicate['frame_id'],
                    "predicate_lemma": predicate['predicate_lemma'],
                    "predicate_span": predicate['span'],
                    "argument_span": arg['span'],
                    "argument": arg['text'],
                    "arg_type": arg['arg_type'],
                    "role_type": arg['role_type'],
                    "arg_head_idx": arg['head_idx']
                })
    onto_df = pd.DataFrame(onto_data)
    return onto_df


def find_question_predicate_idx(predicate, text):
    doc = nlp(text)
    pred_lemma = nlp(predicate)[0].lemma_
    if pred_lemma == "be":
        return -1
    for i, token in enumerate(doc):
        if pred_lemma == token.lemma_:
            return i

    print(f"What is this question again? {predicate}")
    return -1


def find_span(query: str, text: str, compare_lemmas=False):
    text_tokens = text.split()
    query_tokens = query.split()
    if compare_lemmas:
        text_tokens = [t.lemma_.lower() for t in nlp(text)]
        query_tokens = [t.lemma_.lower() for t in nlp(query)]

    for idx, token in enumerate(text_tokens):
        # the worst algorithmic string search,
        # but probably ok (almost linear) in practice
        end_idx = idx + len(query_tokens)
        if query_tokens == text_tokens[idx:end_idx]:
            return f"{idx}:{end_idx}"

    return "-1:-1"


in_path = "./random_sample.csv"
synqg_in_path = "../synqg/random_sample.synqg2.csv"
synqg_out_path = "../synqg/random_sample.synqg2.onto_aligned.csv"
synqg_for_annot_out_path = "../synqg/random_sample.synqg2.onto_aligned.mturk.csv"

names = ['doc_id', 'sent_id', 'predicate', 'question', 'orig_question', 'answer', 'start_char_idx', 'template_name']
in_df = pd.read_csv(in_path)
syn_df = pd.read_csv(synqg_in_path, sep="\t", names=names)

# FOR DEBUGGING.
# syn_df = syn_df[syn_df.doc_id == "nw/wsj/00/wsj_0098"].copy()

onto_df = load_ontonotes(ONTONOTES_DEV_PATH)

sent_df = in_df[['doc_id', 'sent_id',  'text']].drop_duplicates()
syn_df = pd.merge(syn_df, sent_df, on=['doc_id', 'sent_id'])
syn_df2 = syn_df.dropna(subset=['predicate'])
syn_df2['predicate_lemma'] = syn_df2.predicate.apply(lambda s: nlp(s.lower())[0].lemma_)
# not interested in copular questions
syn_df2['predicate_span'] = syn_df2.apply(lambda r: find_span(r.predicate, r.text, compare_lemmas=True), axis="columns")
syn_df2['answer_span'] = syn_df2.apply(lambda r: find_span(r.answer, r.text), axis="columns")
# re-join with the original sample, it had selected specific predicates
# and we don't want other predicates that syn-qg produced over the rest of the sentence
cols = ['doc_id', 'sent_id', 'predicate_span']
syn_df2 = pd.merge(syn_df2, in_df[cols].drop_duplicates(), on=cols)
# filter just to be sure.. should have no effect other than edge cases
syn_df2 = syn_df2[syn_df2.predicate_lemma != "be"].copy()
syn_df2 = syn_df2[syn_df2.predicate_span != "-1:-1"].copy()
syn_df2 = syn_df2[syn_df2.answer_span != "-1:-1"].copy()

# merge with possible arguments from ontonotes.
syn_df3 = pd.merge(syn_df2[cols + ['predicate_lemma', 'question', 'answer', 'answer_span', 'template_name', 'text']],
                   onto_df[cols + ['argument_span', 'sense_id', 'role_type']], on=cols)
syn_df3['iou_score'] = syn_df3.apply(lambda r: iou(parse_span(r.answer_span), parse_span(r.argument_span)),
                                     axis="columns")
syn_df3 = syn_df3[syn_df3.iou_score >= 0.5].copy()
# group and get argument with highest IOU.

syn_df3.sort_values("iou_score", ascending=False, inplace=True)
syn_df3 = syn_df3.groupby(cols + ['question', 'answer_span']).head(1)

# review the questions in a different file
syn_df3 = syn_df3[['doc_id', 'sent_id', 'predicate_lemma', 'sense_id',
                   'question', 'answer', 'predicate_span', 'answer_span',
                   'argument_span', 'role_type', 'text']].copy()
syn_df3.sort_values(['doc_id', 'sent_id', 'predicate_span'], inplace=True)
syn_df3.to_csv(synqg_out_path, sep="\t", index=False, encoding="utf-8")

syn_df4 = syn_df3.groupby(['doc_id', 'sent_id', 'predicate_span', 'text']).agg({'question': list, "role_type": list}).reset_index()
syn_df4['roles'] = syn_df4.role_type.apply(lambda role_types: [f"B{i}-{actual_role}" for i, actual_role in enumerate(role_types)])
syn_df4.drop(columns=['role_type'], inplace=True)
syn_df4.question = syn_df4.question.apply("~!~".join)
syn_df4.rename(columns={"question": "questions"}, inplace=True)
syn_df4.roles = syn_df4.roles.apply("~!~".join)
join_cols = ['doc_id', 'sent_id', 'predicate_span']
# this is taking care of the role descriptions
# but it also filters out all of the added predicates that syn-qg produced but weren't
# in the original sample
syn_df4 = pd.merge(syn_df4, in_df[join_cols + ['role_descriptions']].drop_duplicates(), on=join_cols)
syn_df4 = syn_df4[['doc_id', 'sent_id', 'questions', 'roles', 'predicate_span', 'text', 'role_descriptions']].copy()
syn_df4['experiment_type'] = 'synqg'
syn_df4.to_csv(synqg_for_annot_out_path, index=False, encoding="utf-8")





