import os
from argparse import ArgumentParser
from collections import Counter
from glob import glob
from typing import List, Dict, Tuple
import pandas as pd
from jsonlines import jsonlines
from nltk import Tree
from spacy.tokens import Doc
from tqdm import tqdm
import spacy

if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

from lemma_utils import parse_span, get_spacy_doc, find_argument_head_with_spacy_doc
from ontonotes import Ontonotes, OntonotesSentence


def arg_from_current(start, end, role, tokens):
    return {'span': f"{start}:{end}",
            'text': " ".join(tokens[start:end]),
            'arg_type': 'explicit',
            'role_type': role}


def extract_arguments_from_onto_tags(arg_tags: List[str], onto_sent: OntonotesSentence):
    arguments = []
    current_start = -1
    current_role = None
    tokens = onto_sent.words
    for idx, arg_tag in enumerate(arg_tags):
        if arg_tag == "O":
            if current_role:
                arg = arg_from_current(current_start, idx, current_role, tokens)
                arguments.append(arg)
            current_start, current_role = -1, None
            continue
        bio_prefix, role = arg_tag.split("-", maxsplit=1)

        # role = role.replace("ARG", "A").replace("R-", "").replace("C-", "")
        role = role.replace("ARG", "A")
        if bio_prefix == "B":
            if current_role:
                arg = arg_from_current(current_start, idx, current_role, tokens)
                arguments.append(arg)
            current_start, current_role = idx, role
        elif bio_prefix == "I" and current_role != role:
            raise ValueError("Roles are different!")
    if current_role:
        arg = arg_from_current(current_start, len(arg_tags), current_role, tokens)
        arguments.append(arg)

    for arg in arguments:
        arg_span = parse_span(arg['span'])
        label = get_span_pos(arg_span, onto_sent)
        arg['pos'] = label
    return arguments


def parse_frame(pred_lemma, frame_id, arg_tags, onto_sent):
    arguments = extract_arguments_from_onto_tags(arg_tags, onto_sent)
    predicate = next(filter(lambda a: a['role_type'] == "V", arguments))
    pred_idx = arg_tags.index('B-V')
    arguments.remove(predicate)
    del predicate['role_type']
    del predicate['arg_type']
    predicate['pos'] = onto_sent.pos_tags[pred_idx]
    predicate['lemma'] = pred_lemma
    predicate['frame_id'] = frame_id
    for arg in arguments:
        arg['position'] = 'same_sent'
        arg['arg_type'] = 'explicit'
    frame = {'predicate': predicate, 'arguments': arguments}
    return frame


def get_span_pos(span: Tuple[int, int], onto_sent: OntonotesSentence):
    start, end = span
    if not onto_sent.parse_tree:
        return "N/A"

    tree_pos = onto_sent.parse_tree.treeposition_spanning_leaves(start, end)
    if start == (end - 1):
        # The string is in a leaf node that contains only a string,
        # go back to the parent terminal node to get the syntactic label
        tree_pos = tree_pos[:-1]
    tree_node: Tree = onto_sent.parse_tree[tree_pos]
    pos_label = tree_node.label()
    return pos_label


def parse_coref_mentions(onto_spacy_doc: List[Tuple[OntonotesSentence, Doc]]) -> List[Dict]:
    coref_mentions = []
    for onto_sent, parsed_sentence in onto_spacy_doc:
        for cluster_id, (mention_start, mention_end) in onto_sent.coref_spans:
            mention_end += 1
            mention_text = " ".join(onto_sent.words[mention_start:mention_end])
            pos = get_span_pos((mention_start, mention_end), onto_sent)
            mention_head_idx, mention_dep_rel = find_argument_head_with_spacy_doc((mention_start, mention_end), parsed_sentence)
            mention = {'sent_id': onto_sent.sentence_id,
                       'span': f"{mention_start}:{mention_end}",
                       'pos': pos,
                       "head_idx": mention_head_idx,
                       "dep_rel": mention_dep_rel,
                       'text': mention_text,
                       'cluster_id': cluster_id}
            coref_mentions.append(mention)
    return coref_mentions


def find_arg_mention(arg, coref_mentions):
    for mention in coref_mentions:
        if mention['sent_id'] != arg['sent_id']:
            continue
        if mention['head_idx'] != arg['head_idx']:
            continue
        return mention
    return None


def enrich_arg_with_coref(arg, coref_mentions: List[Dict]):
    the_mention = find_arg_mention(arg, coref_mentions)
    if not the_mention:
        return []
    cls_id = the_mention['cluster_id']
    arg['cluster_id'] = cls_id
    # copy all mentions from my cluster but without me.
    arg_mentions = [dict(m) for m in coref_mentions
                    if m['cluster_id'] == cls_id and m != the_mention]
    for arg_mention in arg_mentions:
        arg_mention['role_type'] = arg['role_type']
        arg_mention['arg_type'] = 'implicit'
        is_same_sent = arg_mention['sent_id'] == arg['sent_id']
        arg_mention['position'] = 'same_sent' if is_same_sent else 'cross_sent'
    return arg_mentions


def yield_sentences_internal(onto_spacy_doc: List[Tuple[OntonotesSentence, Doc]]):
    coref_mentions = parse_coref_mentions(onto_spacy_doc)
    for onto_sent, spacy_doc in onto_spacy_doc:
        doc_id = onto_sent.document_id
        sent = {'text': " ".join(onto_sent.words),
                'doc_id': doc_id,
                'sent_id': onto_sent.sentence_id}
        # This is absurd. Couldn't you provide the predicate lemmas next to the predicates.
        pred_senses = []
        for tok_idx, (word, pred_lemma, pred_frame_id) \
                in enumerate(zip(onto_sent.words,
                                 onto_sent.predicate_lemmas,
                                 onto_sent.predicate_framenet_ids)):
            if not pred_lemma:
                continue
            if not pred_frame_id:
                continue
            arg_tags = [arg_tags for tok, arg_tags in onto_sent.srl_frames if tok == word]
            # remove predicate senses without accompanying tags
            if not arg_tags:
                continue
            pred_senses.append((pred_lemma, pred_frame_id))
        assert len(pred_senses) == len(onto_sent.srl_frames)
        frames = [parse_frame(pred_lemma, frame_id, arg_tags, onto_sent)
                  for (pred_lemma, frame_id), (pred_tokn, arg_tags)
                  in zip(pred_senses, onto_sent.srl_frames)]
        # Let's enrich frames with coref mentions of their arguments
        for frame in frames:
            new_args = []
            frame['predicate']['sent_id'] = onto_sent.sentence_id
            for arg in frame['arguments']:
                arg_span = parse_span(arg['span'])
                arg_head_idx, dep_rel = find_argument_head_with_spacy_doc(arg_span, spacy_doc)
                arg['sent_id'] = onto_sent.sentence_id
                arg['head_idx'] = arg_head_idx
                arg['dep_rel'] = dep_rel
                arg_mentions = enrich_arg_with_coref(arg, coref_mentions)
                new_args.extend(arg_mentions)
            frame['arguments'].extend(new_args)
        sent['frames'] = frames
        yield sent


def reparse_sentences_with_spacy(onto_doc: List[OntonotesSentence]):
    doc_sents = []
    for sent_idx, sent in enumerate(onto_doc):
        spacy_doc = get_spacy_doc(sent.words, nlp)
        doc_sents.append((sent, spacy_doc))
    return doc_sents


def yield_ontonotes_sentences(conll_onto_path: str):
    doc_counter = Counter()
    for onto_doc, partition in yield_ontonotes_documents(conll_onto_path):
        doc_id = onto_doc[0].document_id
        n_sents_in_doc_so_far = doc_counter[doc_id]
        doc_counter[doc_id] += len(onto_doc)
        onto_spacy_sents = reparse_sentences_with_spacy(onto_doc)
        for sent_idx, (onto_sent, spacy_doc) in enumerate(onto_spacy_sents):
            onto_sent.sentence_id = n_sents_in_doc_so_far + sent_idx
        yield from yield_sentences_internal(onto_spacy_sents)


def get_partition(onto_path):
    # Very naive :-)
    if 'test' in onto_path:
        return 'test'
    elif 'development' in onto_path:
        return 'dev'
    else:
        return 'train'


def yield_ontonotes_documents(conll_onto_path: str):
    onto = Ontonotes()
    onto_pattern = "**/*.gold_conll"
    onto_paths = glob(os.path.join(conll_onto_path, onto_pattern), recursive=True)
    for onto_path in tqdm(onto_paths, desc="Parsing PropBank"):
        partition = get_partition(onto_path)
        for onto_doc in onto.dataset_document_iterator(onto_path):
            yield onto_doc, partition


def main(args):

    # Get basic sentences and parse together in one happy file
    # onto_text_path = "./Data/ontonotes/ontonotes.text.tsv"
    # if not os.path.exists(onto_text_path):
    #     doc_counter = Counter()
    #     text_data = []
    #     for onto_doc, partition in (yield_ontonotes_documents(args.conll_onto_path)):
    #         # SOME DOCUMENTS HAVE THE SAME ID!!
    #         # MAKE SURE THEIR SENT IDS DONT MATCH
    #         for sent_idx, onto_sent in enumerate(onto_doc):
    #             doc_id = onto_sent.document_id
    #             actual_sent_id = doc_counter[doc_id]
    #             doc_counter[doc_id] += 1
    #             text_data.append({
    #                 'doc_id': onto_sent.document_id,
    #                 'sent_id': actual_sent_id,
    #                 'text': " ".join(onto_sent.words),
    #                 'pos': " ".join(onto_sent.pos_tags),
    #                 'parse': " ".join(str(onto_sent.parse_tree).split()),
    #                 'partition': partition
    #             })
    #     text_data = pd.DataFrame(text_data).sort_values(['doc_id', 'sent_id'])
    #     text_data.to_csv(onto_text_path, sep="\t", index=False)

    # for part_name in ('train', 'development', 'test'):
    # for part_name in ['development', ]:
    for part_name in ['train', ]:
        onto_path = os.path.join(args.conll_onto_path, "data", part_name)
        ontonotes_sents = list(yield_ontonotes_sentences(onto_path))

        if part_name == "development":
            part_name = "dev"  # :-)
        out_path = os.path.splitext(args.out_path)[0] + f".{part_name}.jsonl"
        with jsonlines.open(out_path, "w") as file_out:
            file_out.write_all(ontonotes_sents)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--conll_onto_path", default="C:\\dev\\ontonotes\\\conll-formatted-ontonotes-5.0")
    ap.add_argument("--out_path", default="./ontonotes/ontonotes.jsonl")
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    main(ap.parse_args())
