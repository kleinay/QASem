import os

from tqdm import tqdm

from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
import pandas as pd
tqdm.pandas()


def find_verb_stem(predicate_lemma):
    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma)
    if not found:
        return predicate_lemma
    return verbs[0]


def make_question(wh, subj, aux, verb_slot, obj, prep, obj2, verb_stem, wiktionary):
    splits = verb_slot.rsplit(maxsplit=1)
    inflection = splits[-1]
    # have been pastParticiple
    pre_verb = ""
    if len(splits) > 1:
        pre_verb = splits[0]

    verb_inflection = wiktionary[verb_stem][inflection] if verb_stem in wiktionary else verb_stem
    question = f"{wh} {aux} {subj} {pre_verb} {verb_inflection} {obj} {prep} {obj2}?"
    question = " ".join(question.replace("_", "").split())
    return question


def load_wiktionary(wik_path):
    # stem  presentSingular3rd  presentParticiple   past        pastParticiple
    # go    goes                going               went        gone

    wik_df = pd.read_csv(wik_path, sep="\t", names=['stem', 'presentSingular3rd', 'presentParticiple', 'past', 'pastParticiple'])
    wiks = wik_df.to_dict(orient="records")
    wik = {wik['stem']: wik for wik in wiks}
    return wik


def normalize_question(s, wiktionary):
    wh = s['wh']
    # Strip the animacy in the question word, turn every who into what.
    if wh.lower() == "who":
        wh = "What"

    is_active = not s['is_passive']
    subj = s['subj']
    if subj != "_":
        subj = "something"
    if is_active:
        aux, verb_slot = ("does", 'stem') if subj == "something" else ("_", "presentSingular3rd")
    else:
        aux, verb_slot = "is", "pastParticiple"

    obj = s['obj']
    if obj != "_":
        obj = "something"
    obj2 = s['obj2']
    prep = s['prep'] or "_"
    lemma = s['predicate_lemma']
    verb_stem = s['verb_form'] if 'verb_form' in s else find_verb_stem(lemma)
    proto_question = make_question(wh, subj, aux, verb_slot, obj, prep, obj2, verb_stem, wiktionary)
    return proto_question








