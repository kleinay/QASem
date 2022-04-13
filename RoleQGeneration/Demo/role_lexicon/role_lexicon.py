import itertools
from typing import Union, List, Optional
import pandas as pd
import logging
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

from common_types import Role, Predicate


class RoleLexicon:
    def __init__(self, entries):
        entries = [Role(predicate=e.get('lemma') or e.get("predicate_lemma"),
                        sense_id=f"{e['sense_id']:02d}",
                        pos=RoleLexicon.normalize_pos(e['pos']),
                        role_type=e['role_type'],
                        role_desc=e['role_desc'], role_set_desc=e['roleset_desc'])
                   for e in entries]
        roleset_map = {}
        all_rolesets_map = {}
        entries = sorted(entries, key=RoleLexicon.role_sort_order)
        self.df = pd.DataFrame(entries)
        for key_items, roleset in itertools.groupby(entries, key=RoleLexicon.roleset_sort_order):
            key = self.to_key(*key_items)
            roleset_map[key] = list(roleset)

        for key_items, roleset in itertools.groupby(entries, key=RoleLexicon.all_rolesets_sort_order):
            key = '.'.join(key_items)
            all_rolesets_map[key] = list(roleset)

        self.roleset_map = roleset_map
        self.all_rolesets_map = all_rolesets_map

    def __getitem__(self, item) -> Union[Role, List[Role]]:
        if len(item) == 4:
            predicate_lemma, sense, pos, role = item[:4]
            return self.get_role(predicate_lemma, sense, pos, role)
        if len(item) == 3:
            predicate_lemma, sense, pos = item[:3]
            return self.get_roleset(predicate_lemma, sense, pos)
        if len(item) == 2:
            predicate, role = item[:2]
            return self.get_role(predicate.lemma, predicate.sense_id, predicate.pos, role)

    def get_roleset(self, predicate: Predicate) -> Optional[List[Role]]:
        return self.get_roleset(predicate.lemma, predicate.sense_id, predicate.pos)

    def get_roleset(self, predicate: str, sense_id: str, pos: str) -> Optional[List[Role]]:
        key = self.to_key(predicate, sense_id, pos)
        return self.roleset_map.get(key)

    def get_all_rolesets(self, predicate: str):
        df = self.df
        pred_df = df[df.lemma == predicate].copy()
        return pred_df

    def get_all_rolesets2(self, predicate: str, pos: str = "v") -> Optional[List[Role]]:
        pos = RoleLexicon.normalize_pos(pos[0].lower())
        key = f"{predicate}.{pos}"
        all_rolesets = self.all_rolesets_map.get(key)
        # first fallback, try the same predicate lemma with a verb roleset
        # without consulting qanom
        if not all_rolesets and pos == 'n':
            new_key = f"{predicate}.v"
            all_rolesets = self.all_rolesets_map.get(new_key)
        # second fallback: use qanom to find the verb form for this noun.
        if not all_rolesets and pos == 'n':
            verbs, found = get_verb_forms_from_lexical_resources(predicate, filter_distant=True)
            if found:
                for verb in verbs:
                    new_key = f"{verb}.v"
                    all_rolesets = self.all_rolesets_map.get(new_key, [])
        return all_rolesets

    def get_role(self, predicate: str, sense_id: str, pos: str, role: str) -> Optional[Role]:
        roleset = self.get_roleset(predicate, sense_id, pos)
        if not roleset:
            return None

        roles = [r for r in roleset if r.role_type == role]
        return next(iter(roles), None)

    @staticmethod
    def normalize_pos(pos: str) -> str:
        return pos[0].lower()

    def to_key(self, *args) -> str:
        predicate_lemma, sense_id, pos = args[:3]
        pos = RoleLexicon.normalize_pos(pos)
        if isinstance(sense_id, int):
            sense_id = f"{sense_id:02d}"
        return f"{predicate_lemma}.{sense_id}{pos}"

    @staticmethod
    def role_sort_order(entry: Role):
        e = entry
        return e.predicate, e.sense_id, e.pos, e.role_type

    @staticmethod
    def roleset_sort_order(entry: Role):
        e = entry
        return e.predicate, e.sense_id, e.pos

    @staticmethod
    def all_rolesets_sort_order(entry: Role):
        e = entry
        return e.predicate, e.pos

    @classmethod
    def from_file(cls, lexicon_path: str) -> 'RoleLexicon':
        if lexicon_path.endswith(".tsv"):

            df = pd.read_csv(lexicon_path, sep="\t")
            roles = df.to_dict(orient="records")
            return cls(roles)
        else:
            raise NotImplementedError(f"unrecognized extension: {lexicon_path}")

