from dataclasses import dataclass
from typing import Tuple, Dict


CORE_ROLES = [f"A{i}" for i in range(5)]

ADJUNCT_ROLES = {"Am", "AM", "AM-TMP", "AM-LOC", "AM-MNR",
                 "AM-CAU", "AM-PRD", "AM-DIR", "AM-EXT",
                 "AM-PRD", "AM-PNC", "AM-REC", "AM-DIS", "AM-ADV"}


ADJUNCT_DESC = {
    "AM-TMP": "Temporal",
    "AM-LOC": "Locative",
    "AM-MNR": "Manner",
    "AM-CAU": "Cause",
    "AM-PRD": "Predicative",
    "AM-DIR": "Direction",
    "AM-PNC": "Purpose",  # Purpose, not Cause...
    "AM-EXT": "Extent",
    "AM-REC": "Reciprocal",
    "AM-DIS": "Discourse",
    "AM-ADV": "Adverbial",
}

NO_SPAN = "-1:-1"


def parse_span(span_: str) -> Tuple[int, int]:
    s, t = span_.split(":")
    return int(s), int(t)


@dataclass
class Role:
    predicate: str
    sense_id: str
    pos: str
    role_type: str
    role_desc: str
    role_set_desc: str


@dataclass
class Argument:
    _span: str
    text: str
    arg_type: str
    role_type: str
    sent_id: str

    @property
    def start(self):
        return self.span[0]

    @property
    def end(self):
        return parse_span(self._span)[1]

    @property
    def span(self) -> Tuple[int, int]:
        return parse_span(self._span)

    @classmethod
    def from_dict(cls, d):
        return cls(_span=d['span'],
                   text=d['text'],
                   role_type=d['role_type'],
                   arg_type=d.get('arg_type', "explicit"),
                   sent_id=d.get('sent_id', "-1"))


@dataclass
class Predicate:
    _span: str
    text: str
    lemma: str
    sense_id: str
    pos: str
    sent_id: str

    @property
    def start(self) -> int:
        return self.span[0]

    @property
    def end(self) -> int:
        return self.span[1]

    @property
    def span(self) -> Tuple[int, int]:
        return parse_span(self._span)

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(d['span'], d['text'],
                   d.get('lemma') or d.get('predicate_lemma'),
                   d.get('sense_id') or d.get("frame_id", "-1"),
                   d.get('pos', "v"),
                   d.get('sent_id'))
