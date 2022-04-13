from typing import List
import sys
import jsonlines

from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

RoleSet = List[str]


def find_verb_stem(predicate_lemma):
    verbs, found = get_verb_forms_from_lexical_resources(predicate_lemma, filter_distant=True)
    if not found:
        return predicate_lemma
    return verbs[0]


class PropBankRoleFinder:
    def __init__(self, predicates):
        # {"predicate": "abduct", "pos": "v",
        #     "role_sets": [
        #           {"sense_id": 1, "roles": [{"type": "ARG0", "desc": "agent"},
        #                                     {"type": "ARG1", "desc": "person kidnapped"}]}]}
        self.pred_to_rolesets = {f"{prd['predicate']}-{prd['pos']}": prd['role_sets']
                                 for prd in predicates}

    def get_roles(self, predicate_lemma: str, pos="v", sense=-1) -> RoleSet:
        rolesets = self.pred_to_rolesets.get(f"{predicate_lemma}-{pos}", [])
        if not rolesets and pos == "n":
            rolesets = self.pred_to_rolesets.get(f"{predicate_lemma}-v", [])
        if not rolesets and pos == "n":
            # still no rolesets, let's look at the verbal forms.
            verb_form = find_verb_stem(predicate_lemma)
            rolesets = self.pred_to_rolesets.get(f"{verb_form}-v", [])
        if not rolesets:
            return ["A0", "A1", "A2", "A3"]
        if sense >= 0:
            rolesets = [r for r in rolesets if r['sense_id'] == sense]
        # combine all possible roles from all predicate-senses
        roles = [(role['type'], role['desc']) for role_set in rolesets for role in role_set['roles']]
        return sorted(set(roles))

    @classmethod
    def from_framefile(cls, path_to_frame_file):
        with jsonlines.open(path_to_frame_file) as file_in:
            predicates = list(file_in)

        return cls(predicates)
