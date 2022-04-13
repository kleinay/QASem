from typing import Dict, Any, List

from collections import defaultdict
import spacy
from spacy.tokens import Doc, Token

from .question import TemplateQuestionGenerator
from .role import PropBankRoleFinder
from .answer import HuggingFaceQAPredictor, Span

CONTENT_POS = ["NOUN", "VERB"]
ADJUNCTS = ["AM-TMP", "AM-LOC", 'AM-ADV', 'AM-MOD']
MAIN_ROLES = ["A0", "A1", "A2", "A3", "A4", "A5"]
DEFAULT_ROLES = MAIN_ROLES + ADJUNCTS

NO_ANSWER: Span = (0, 0)
INVALID_ANSWER: Span = (-1, -1)


def is_null_or_invalid(span: Span):
    return span == NO_ANSWER or span == INVALID_ANSWER


def get_answer(span: Span, tokens: List[str]):
    if span == INVALID_ANSWER:
        return "[INVALID]"
    elif span == NO_ANSWER:
        return "[NO_ANSWER]"
    else:
        start, end = span
        return " ".join(tokens[start: end])


class SemanticRoleAsQAParser:

    def __init__(self,
                 ques_gen_predictor: TemplateQuestionGenerator,
                 role_finder: PropBankRoleFinder,
                 answer_solver: HuggingFaceQAPredictor):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.ques_generator = ques_gen_predictor
        self.role_finder = role_finder
        self.answer_solver = answer_solver

    def yield_qas_from_predicate(self, tokens: List[str], predicate_indices: List[int]):
        for pred_idx in predicate_indices:
            pred_token = self.get_predicate_spacy_token(tokens, pred_idx)
            the_roles = self.role_finder.get_roles(pred_token.lemma_)
            for role in the_roles:
                questions = self.ques_generator.generate(pred_token.lemma_, role)
                if not questions:
                    continue
                for question in questions:
                    yield pred_idx, question, role

    def parse(self, sample: Dict[str, Any]):
        tokens = sample['text'].split()
        if 'predicate_indices' in sample:
            predicate_indices = sample['predicate_indices']
        else:
            predicate_indices = self.detect_predicates(tokens)

        search_space = sample.get('search_space', [])
        spans_for_role = defaultdict(set)
        qa_pairs = []
        for pred_idx, question, role in self.yield_qas_from_predicate(tokens, predicate_indices):
            span = self.answer_solver.predict(question, tokens, (pred_idx, pred_idx + 1), search_space)
            # We have already predicted this span for some other question for this role
            if span in spans_for_role[(pred_idx, role)]:
                continue
            spans_for_role[(pred_idx, role)].add(span)
            answer = get_answer(span, tokens)
            qa_pair = {'role_type': role,
                       'predicate_span': f"{pred_idx}:{pred_idx+1}",
                       'question': question,
                       'predicted_answer_span': f"{span[0]}:{span[1]}",
                       'predicted_answer': answer,
                       'text': ' '.join(tokens)}

            if 'doc_id' in sample:
                qa_pair['doc_id'] = sample['doc_id']
                qa_pair['sent_id'] = sample['sent_id']
            qa_pairs.append(qa_pair)

        return qa_pairs

    def detect_predicates(self, tokens):
        doc = Doc(words=tokens, vocab=self.nlp.vocab)
        indices = [token.id for token in doc
                   if token.tag_ in CONTENT_POS]
        return indices

    def get_predicate_spacy_token(self, tokens: List[str], predicate_idx: int) -> Token:
        doc = spacy.tokens.Doc(words=tokens, vocab=self.nlp.vocab)
        doc = self.nlp.tagger(doc)
        token = doc[predicate_idx]
        return token
