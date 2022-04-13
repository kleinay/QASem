import os
from argparse import ArgumentParser

import pandas as pd

VERB_LABELS = [
    'stem',
    'pastParticiple',
    'presentSingular3rd',
    'past',
    'being pastParticiple',
    'presentParticiple',
    'be pastParticiple',
    'been pastParticiple',
    'be presentParticiple',
    'have pastParticiple',
    'been presentParticiple',
    'have been pastParticiple',
    'not stem',
    'not be pastParticiple',
    'have been presentParticiple',
    'not be presentParticiple',
    'not have pastParticiple',
    'not have been pastParticiple',
    'not have been presentParticiple',
]


question_slots = ['wh', 'subj', 'obj', 'obj2', 'aux', 'prep',
                  'is_passive', 'is_negated', 'verb_prefix', "verb_slot_inflection",]
cols = ['question', 'answer_range', 'answer', 'target_idx', 'noun', 'verb_form', 'sentence', 'qasrl_id']
cols_out = ['question', 'predicate', 'gold_answers', 'verb_form', 'doc_id',
            'gold_answer_spans', 'predicate_span',
            *question_slots[:-2], 'verb', 'text']


def fix_verb_slot(verb):
    verb = verb.replace("~!~", " ")
    for s in ["Past", "Stem", "Present"]:
        r = s[0].lower() + s[1:]
        verb = verb.replace(s, r)
    # THIS BUG IS STRANGE. WHY THE PARTICIPLE WAS REMOVED?
    if verb not in VERB_LABELS and verb + "Participle" in VERB_LABELS:
        verb += "Participle"
    return verb


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--qanom_dir", required=True, help="C:\\dev\\qanom")
    args = ap.parse_args()
    for split in ('dev', 'train'):
        qanom_path = os.path.join(args.qanom_dir, f"{split}.csv")
        df = pd.read_csv(qanom_path, usecols=cols + question_slots)
        # To be aligned with QASRL Bank data, build the "verb" field from
        # the prefix and inflection slots. However, pastParticiple is marked as PAST.
        # Can't tell why
        df.verb_prefix.fillna("", inplace=True)
        df.verb_slot_inflection.fillna("", inplace=True)
        df['verb'] = df.apply(lambda r: f"{r.verb_prefix} {r.verb_slot_inflection}".strip(), axis="columns")
        df.verb = df.verb.apply(fix_verb_slot)
        df = df.dropna(subset=cols)
        df.rename(columns={'sentence': 'text', 'answer_range': 'gold_answer_spans', 'qasrl_id': 'doc_id',
                           'answer': 'gold_answers', 'noun': 'predicate', 'target_idx': 'predicate_span'}, inplace=True)
        df.predicate_span = df.predicate_span.apply(lambda idx: f"{idx}:{idx+1}")
        df[cols_out].to_csv(f"./qasrl/qanom.{split}.tsv", index=False, encoding="utf-8", sep="\t")



