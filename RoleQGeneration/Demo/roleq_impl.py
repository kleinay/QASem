import itertools
import logging
import os
import pandas as pd
from argparse import ArgumentParser

import lemma_utils
from common_types import Role


from typing import Tuple, Union, List, Optional, Dict
import spacy
from demo.role_lexicon.role_lexicon import RoleLexicon
import csv
from collections import defaultdict
from question_translation import QuestionTranslator
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

logger = logging.getLogger(__file__)


def read_covered_predicates(covered_path):
    infile = csv.reader(open(covered_path), delimiter='\t')
    # predicate : pos  : mapping
    covered_predicates = defaultdict(lambda: defaultdict(lambda: ''))
    for row in infile:
        covered_predicates[row[0]][row[1]] = row[2]
    return covered_predicates


def get_proto_question_dict(proto_path):
    df = pd.read_csv(proto_path, sep="\t")
    df.sense_id = df.sense_id.apply(lambda s: f"{s:02d}")
    role_cols = ['verb_form', 'sense_id', 'role_type']
    best_proto_df = df.sort_values(by=["squad_f1"], ascending=False).groupby(role_cols).head(1)
    best_protos = best_proto_df.to_dict(orient='records')
    proto_dict = {
        f"{r['verb_form']}{r['sense_id']}{r['role_type']}": r['proto_question']
        for r in best_protos
    }
    rolesets_df = best_proto_df[['verb_form', 'sense_id']].drop_duplicates()
    # verb_forms = set(sorted([r['verb_form'] for r in best_protos]))
    return proto_dict, rolesets_df


def roleset_simple_agg(role: Role):
    return role.predicate, role.sense_id


def get_verb_form(noun) -> Optional[str]:
    verbs, found = get_verb_forms_from_lexical_resources(noun)
    if not found:
        return None
    return next(verbs, None)


class RoleQDemo:
    def __init__(self, proto_dict, rolesets_df, lex, q_translator, nlp):
        self.proto_dict = proto_dict
        self.covered_rolesets_df = rolesets_df
        self.covered_verb_forms = set(rolesets_df['verb_form'].unique())
        self.lex = lex
        self.q_translator = q_translator
        self.nlp = nlp

    def get_covered_verb_form(self, token) -> List[str]:
        lemma = token.lemma_
        pos = token.pos_
        if token.pos_ not in ['VERB', 'NOUN']:
            return []
        if lemma in self.covered_verb_forms:
            return [lemma]
        if pos.upper() == "VERB":
            return []
        # Convert nouns to their verbalized forms...
        verbs, found = get_verb_forms_from_lexical_resources(lemma)
        if not found:
            return []
        covered_verbs = list(set(verbs).intersection(self.covered_verb_forms))
        return covered_verbs

    def analyze(self, text: str) -> Dict:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        indices = []
        lemmas = []
        for token in doc:
            if token.lemma_ in ("be", 'have'):
                continue
            covered_verb_forms = self.get_covered_verb_form(token)
            if covered_verb_forms:
                indices.append(token.i)
            lemmas.append(token.lemma_ or "-")
        return {"tokens": tokens, "predicate_indices": indices, "lemmas": lemmas}

    def get_rolesets(self, predicate_idx: int, tokens: List[str]) -> List:
        doc = lemma_utils.get_spacy_doc(tokens, self.nlp)
        pred_token = doc[predicate_idx]
        covered_verb_forms = self.get_covered_verb_form(pred_token)
        # these sense ids have prototype questions.
        covered_rolesets_df = self.covered_rolesets_df[self.covered_rolesets_df.verb_form.isin(covered_verb_forms)].copy()
        #  An edge case - flight.01.n.A4 would never be found because fly.01.v.A4 doesn't exist.
        my_df = pd.merge(self.lex.df, covered_rolesets_df,
                         left_on=['predicate', 'sense_id'],
                         right_on=['verb_form', 'sense_id'])
        my_df = my_df[['predicate', 'pos', 'sense_id', 'role_set_desc']].drop_duplicates()
        my_df.rename(columns={'predicate': 'lemma'}, inplace=True)
        my_rolesets = my_df.to_dict(orient="records")
        return my_rolesets

    def get_questions(self, lemma: str, pos: str, sense_id: str, predicate_idx: int, tokens: List[str]) -> List:
        text = ' '.join(tokens)
        all_roles = self.lex.get_roleset(lemma, sense_id, pos)
        predicate_span = str(predicate_idx) + ':' + str(predicate_idx + 1)
        samples = []
        actual_roles = []
        for role in all_roles:
            proto_question = self.proto_dict.get(lemma + sense_id + role.role_type)
            if not proto_question:
                continue
            samples.append({'proto_question': proto_question,
                            'predicate_lemma': lemma,
                            'predicate_span': predicate_span,
                            'text': text})
            actual_roles.append(role)
        contextualized_questions = self.q_translator.predict(samples)
        out = [
            {"role_type": role.role_type, "role_desc": role.role_desc,
             "questions": [{"prototype": sample['proto_question'],
                            "contextualized": full_question}]
             }
            for role, sample, full_question
            in zip(actual_roles, samples, contextualized_questions)]

        return out

    def generate(self, prototype: str, predicate_idx: int, tokens: List[str], selected_lemma) -> Dict:
        predicate_span = f"{predicate_idx}:{predicate_idx + 1}"
        text = ' '.join(tokens)
        samples = [{'proto_question': prototype, 'predicate_lemma': selected_lemma,
                    'predicate_span': predicate_span,
                    'text': text}]
        contextualized_question = self.q_translator.predict(samples)[0]
        return {"contextualized_question": contextualized_question}


def setup_roleqs(args):
    logger.info(f"Loading prototype questions from: {args.proto_path}")
    proto_dict, rolesets_df = get_proto_question_dict(args.proto_path)
    logger.info(f"Loading  question translation model from: {args.trans_model}")
    q_translator = QuestionTranslator.from_pretrained(args.trans_model, device_id=args.device_id)
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "tetxtcat"])  # "tok2vec", "attribute_ruler"])
    logger.info(f"Loading lexicon from: f{args.lex_path}")
    lex = RoleLexicon.from_file(args.lex_path)
    logger.info(f"Loaded: {lex.df.shape[0]} roles")
    roleqdemo = RoleQDemo(proto_dict, rolesets_df, lex, q_translator, nlp)
    return roleqdemo


# def main():
#     transformation_model_path = "/home/nlp/pyatkiv/workspace/transformers/examples/seq2seq/question_transformation_grammar_corrected_who/"
#     ap = ArgumentParser()
#     ap.add_argument("--device_id", type=int, default=0)
#     ap.add_argument("--trans_model", default=transformation_model_path)
#     ap.add_argument("--proto_path", default="./resources/qasrl.prototype_accuracy.ontonotes.tsv")
#     ap.add_argument("--lex_path", default="./role_lexicon/predicate_roles.ontonotes.tsv")
#
#     ap.add_argument("--device_id", type=int, default=0)
#     args = ap.parse_args()
#
#     roleqdemo = setup_roleqs(args)
#     indices = roleqdemo.analyze('John sold a pen to Mary.')
#     print(indices)
#     rolesets = roleqdemo.get_rolesets(1, ['John', 'sell', 'a', 'pen', 'to', 'Mary'], ['NOUN', 'VERB'])
#     print(rolesets)
#     questions = roleqdemo.get_questions("sell", "v", "01", 1, ['John', 'sell', 'a', 'pen', 'to', 'Mary'])
#     print(questions)
#     filled = roleqdemo.generate("What sells something?", 1, ['John', 'sell', 'a', 'pen', 'to', 'Mary'], "sell")
#     print(filled)


# if __name__ == "__main__":
#     main()
