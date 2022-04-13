from typing import List, Tuple
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner'])
REPLACE_TOKENS = {"-LRB-": "(","-RRB-": ")", "-LSB-":"[", "-RSB-": "]", "-LCB-":"{", "-RCB-": "}"}


def get_lemma_pos(tokens: List[str], token_idx: int, local_nlp: spacy.language.Language = nlp):
    parse = spacy.tokens.Doc(words=tokens, vocab=local_nlp.vocab)
    for name, component in local_nlp.pipeline:
        if name not in ['tok2vec', 'lemmatizer', 'tagger', 'attribute_ruler']:
            continue
        parse = component(parse)
    # tagger = local_nlp.get_pipe("tagger")
    # parse = tagger(parse)
    token = parse[token_idx]
    return token.lemma_, token.tag_


def get_lemma(tokens: List[str], token_idx: int, local_nlp: spacy.language.Language = nlp):
    lemma, pos = get_lemma_pos(tokens, token_idx, local_nlp)
    return lemma


def get_verb_form(lemma):
    verb_forms, is_ok = get_verb_forms_from_lexical_resources(lemma)
    return verb_forms[0]


def parse_span(span: str):
    start, end = span.split(":")
    return int(start), int(end)


def get_spacy_doc(tokens: List[str], local_nlp=None):
    if not local_nlp:
        local_nlp = nlp
    if not local_nlp:
        raise RuntimeError("You should supply an instance of spacy loaded language")
    doc = spacy.tokens.Doc(words=tokens, vocab=local_nlp.vocab)
    # Apply the tagger and parser pipelines...
    for name, pipeline in local_nlp.pipeline:
        doc = pipeline(doc)
    return doc


def find_argument_head(span: Tuple[int, int], arg_sentence: List[str]) -> int:
    doc = get_spacy_doc(arg_sentence)
    return find_argument_head_with_spacy_doc(span, doc)[0]


def find_argument_head_with_spacy_doc(span: Tuple[int, int], parsed_sentence: spacy.tokens.Doc) -> Tuple[int, str]:
    arg_span = parsed_sentence[span[0]:span[1]]
    # https://spacy.io/api/span#root
    # The token with the shortest path to the root of the sentence (or the root itself).
    # If multiple tokens are equally high in the tree, the first token is taken.
    arg_root = arg_span.root
    # if this is a preposition, try to find the object of the preposition this span is made of.
    arg_children = list(arg_root.children)
    if arg_root.dep_ == "prep" and len(arg_children) == 1:
        first_child = arg_children[0]
        if first_child.dep_ == "pobj":
            arg_root = first_child

    head_idx = arg_root.i
    dep_rel = arg_root.dep_
    return head_idx, dep_rel


def fixup_token(token: str):
    if token in REPLACE_TOKENS:
        return REPLACE_TOKENS[token]
    return token


def should_remove(token, tag) -> bool:
    return tag == "-NONE-"


def update_span_indices(span_obj, index2new, clean_text):
    new_span_obj = dict(span_obj)
    begin, end = span_obj["span"].split(":")
    new_begin, new_end = index2new[int(begin)], index2new[int(end)]
    # save a debug message if the token was completely removed
    if new_begin == new_end:
        new_span_obj['debug'] = span_obj['text']
    new_span_obj["span"] = f"{new_begin}:{new_end}"
    new_span_obj["text"] = " ".join(clean_text[new_begin:new_end])
    return new_span_obj


def clean_frame(frame, tagged_sents: List[List[Tuple[str, str]]]):
    args = frame["arguments"]
    index2new = {}  # every span boundary will have a key, where its value will be its new boundary index.
    # So ex. old = 13:16, and there will be a key 13 with new value 12, and key for 16 with value 14.
    for arg in args + [frame["predicate"]]:
        begin, end = arg["span"].split(":")
        begin, end = int(begin), int(end)
        index2new[begin] = begin  # initialize with the same spans
        index2new[end] = end

    sent_id = frame['sent_id']
    if isinstance(sent_id, str):
        start_sent_id, end_sent_id = parse_span(frame['sent_id'])
    else:
        start_sent_id, end_sent_id = sent_id, sent_id+1
    frame_sents = tagged_sents[start_sent_id: end_sent_id]
    frame_tag_tokens = [tok for sent in frame_sents for tok in sent]
    n_orig_tokens = len(frame_tag_tokens)

    tokens_removed = []
    clean_text = []
    clean_obj = dict(frame)
    for ind, (token, tag) in enumerate(frame_tag_tokens):
        token = fixup_token(token)
        if should_remove(token, tag):
            tokens_removed.append(ind)
        else:
            clean_text.append(token)

        # if curr index is one of our argument span indices,
        # then we update it with the # of tokens removed till that point
        if int(ind) in index2new:
            index2new[ind] = index2new[ind] - len(tokens_removed)
            if should_remove(token, tag):
                # since the bad token is in the beginning of a span
                index2new[ind] += 1
    # For edge cases where an argument appears at the end of the sentence
    # we have to update its end index (that is exclusive of the sentence bounds)
    # to be the new sentence boundary
    index2new[n_orig_tokens] = len(clean_text)

    # update spans with new indices and argument text
    args = frame["arguments"]
    clean_obj['predicate'] = update_span_indices(frame['predicate'], index2new, clean_text)
    clean_obj['arguments'] = [update_span_indices(arg, index2new, clean_text)
                              for arg in args]

    clean_text = " ".join(clean_text)
    clean_obj["text"] = clean_text
    return clean_obj
