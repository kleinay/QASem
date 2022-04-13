from collections import defaultdict, Counter
from typing import Dict, List

import pandas as pd
import spacy
import codecs
import re


class TemplateQuestionGenerator:
    def __init__(self, pred_to_questions: Dict[str, List[Dict]], proto_question_dict=None):
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.pred_to_questions = pred_to_questions
        self.proto_question_dict = proto_question_dict or {}
        self.not_found_counter = Counter()

    def generate(self, predicate_lemma: str, role: str):
        question_roles = self.pred_to_questions.get(predicate_lemma, [])
        matching_questions = [q['question'] for q in question_roles
                              if q['role'] == role]
        if not matching_questions and role in self.proto_question_dict:
            self.not_found_counter[(predicate_lemma, role)] += 1
            question = self.proto_question_dict[role]
            question = re.sub('PREDICATE', predicate_lemma, question)
            matching_questions.append(question)
        if not matching_questions:
            matching_questions.append('PREDICATE_OR_ROLE_NOT_COVERED')
        return matching_questions

    @classmethod
    def from_path(cls, questions_path, proto_questions_path=None):
        # This is a stub
        role_questions = pd.read_csv(questions_path, sep="\t").to_dict(orient="records")
        pred_to_questions = cls.get_predicate_question_map(role_questions)
        proto_question_map = None
        if proto_questions_path:
            proto_question_map = cls.get_proto_questions(proto_questions_path)
        return cls(pred_to_questions, proto_question_map)

    @staticmethod
    def get_predicate_question_map(role_questions) -> Dict[str, List[Dict]]:
        # predicate : (question, role, role_description)
        pred_q_dict = defaultdict(list)
        for rec in role_questions:
            question_role = {key: rec[key] for key in ('question', 'role')}
            pred_q_dict[rec['predicate']].append(question_role)

        return pred_q_dict

    @staticmethod
    def get_proto_questions(proto_questions_path: str):
        # role: proto_question
        protoq_dict = defaultdict(lambda: '')
        # This external dependency should be injected!
        # infile = codecs.open('scripts/question_role_prototypes.tsv', 'r')
        infile = codecs.open(proto_questions_path, 'r')
        for line in infile.readlines():
            line = line.split('\t')
            protoq_dict[line[0]] = line[1]
        return dict(protoq_dict)
