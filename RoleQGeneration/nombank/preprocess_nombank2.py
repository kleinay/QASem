import os
from argparse import ArgumentParser, Namespace
from typing import Tuple, List

from jsonlines import jsonlines
from tqdm import tqdm
from collections import defaultdict

from lemma_utils import clean_frame
from nltk.corpus.reader import BracketParseCorpusReader
from nltk import Tree

from ontonotes.ontonotes import Ontonotes

WSJ_FILE_PATTERN = ".*/WSJ_.*\.MRG"
EMPTY_OR_WH = {"-NONE-", "WDT", "WP", "WP$"}


def parse_simple_constituent(constituent: str, tree: Tree) -> Tuple[int, int]:
    """
    @param constituent:
    @param tree: The sentence
    @return:
    """
    is_complex = "-" in constituent or "*" in constituent
    if is_complex:
        raise SyntaxError(f"Constituent should be a simple tree pointer: {constituent}")
    start, height = constituent.split(":")
    start, height = int(start), int(height)
    # an array of child indices at every tree level beginning with the root.
    # At index (i) go to child with index leaf_pos[i] for the next node
    # until you reach this leaf node.
    leaf_pos = tree.leaf_treeposition(start)
    # add 1 to height to get the leaf node (otherwise we get directly the string)
    height += 1
    # we want to take the node that is 'height' levels above the leaf node.
    node_pos = leaf_pos[:-height]
    node = tree[node_pos]
    # remove empty categories and single token nodes with
    # WH-Determiner (WDT) WH-pronoun (WP) and WH Possesive (WP$)
    # There are unhandled edge cases where the node contains multiple EMPTY leaves.
    is_single_token = len(node.leaves()) == 1
    leaf_label = tree[leaf_pos[:-1]].label()
    is_empty_or_wh = leaf_label in EMPTY_OR_WH
    should_remove = (is_single_token and is_empty_or_wh)
    if should_remove:
        return -1, -1

    phrase = node.leaves()
    end = start + len(phrase)
    # VALIDATION FOR SOME WEIRD EDGE CASES
    n_tokens = len(tree.leaves())
    if end > n_tokens:
        print(f"Constituent is out of sentence boundaries: {constituent}, "
              f"start: {tree[leaf_pos[:-1]]} \n"
              f" tree: {tree} ")
        return -1, -1
    return start, end


def coalesce_contiguous_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_spans = []
    if not spans:
        return new_spans

    curr_span = spans[0]
    for span in spans[1:]:
        is_consecutive = curr_span[1] == span[0]
        if is_consecutive:
            curr_span = curr_span[0], span[1]
        else:
            new_spans.append(curr_span)
            curr_span = span
    new_spans.append(curr_span)
    return new_spans


def parse_role(role: str):
    '''
    @param role:
    @return:
    '''

    new_role = role.replace("ARG", "A")

    # There seems to be a single annotation mistake in NomBank
    if new_role == "rel-REF":
        new_role = "rel"
    return new_role


def parse_members(members, tree: Tree):
    for member in members:
        pointer, role = member.split("-", maxsplit=1)
        # A feature label can consist of one or more parts separated by
        # hyphens. The initial part of the label indicates whether the item is a
        # REL (the predicate), SUPPORT (a support item), ARGM (an adjunct) or an
        # argument (ARG0 through ARG9). Each subsequent piece of the label is a
        # function tag. A full list of these is given in the specifications. For
        # example, ARGM-TMP is a temporal adjunct where TMP indicates the type
        # of the adjunct. There are also a special set of function tags called
        # hyphen tags {H0, H1, H2, H3, H4, H5, H6, H7, H8, H9}. These are used
        # to divide items that are a single token but containing hyphens "-"
        # and/or forward slashes "/". H0 signifies the segment before the first
        # hyphen or slash and Hn signifies the segment after the nth hyphen or
        # slash. For example, ARG1-H0 can mean that the first segment ("auto")
        # of "auto-salesman" is marked with an ARG1 role. REL-H1 may mean that
        # the second segment ("salesman") is the main predicate.
        role = parse_role(role)

        # For example, in "John made a series of
        # mistakes", "made", "series" and "of" form a chain of lexical items
        # that link "John" as the ARG0 of "mistakes". This use of this notation
        # is not intended to suggest that the support items form a
        # constituent
        if 'Support' in role:
            continue
        # 7:1-rel   9:2*16:0-ARG1   21:0-Support    13:0,14:0-Support   12:0-ARG1-H0-H1
        # According to NomBank documentation, a pointer can have multiple levels:
        # (c) A pointer chain: a list of pointers separated by asterisks. This
        # indicates a coreference chain of empty categories (gaps), relative
        # pronouns and regular constituents. This is normally used if an empty
        # category is local in argument structure.
        # Hopefully, there is no multi-level recursion and
        # we can only have coref-chains --> constituents --> simple pointers
        corefs = pointer.split("*")
        for mention in corefs:
            # (b) A concatenated pointer: a list of pointers separated by
            # commas. This can indicate that while these items do not form a
            # constituent in the Penn Treebank, they do actually form a
            # constituent.  However, there are also some unusual instances where
            # disjoint constituents share an argument position. For example, consider
            # "the George Bush speech", where "the" is the first token (position
            # 0). Here "George Bush" is the ARG0 of "speech". The piece of the
            # proposition representing this would be: "1:0,2:0-ARG0". The
            # concatenated pointer is also used for listing support items when they
            # form a support chain. For example, in "John made a series of
            # mistakes", "made", "series" and "of" form a chain of lexical items
            # that link "John" as the ARG0 of "mistakes".
            constituents = mention.split(",")
            # (a) A simple pointer: two numbers separated by a colon, the first
            # indicating a token position (as above). The second indicates number of
            # levels in the tree above the token level
            spans = [parse_simple_constituent(c, tree) for c in constituents]
            # all the non-informative single token arguments (whose, who, that, which) were removed
            spans = [sp for sp in spans if sp != (-1, -1)]
            spans = coalesce_contiguous_spans(spans)
            for span in spans:
                yield span, role


def get_pos_for_span(span: Tuple[int, int], tree: Tree):
    leaf_pos = tree.leaf_treeposition(span[0])
    leaf_node = tree[leaf_pos[:-1]]
    pos = leaf_node.label()
    return pos


def parse_nominal_frame(members: str, tree: Tree, tokens: List[str]):
    spans_and_roles = list(parse_members(members, tree))
    frame = {"predicate": {}, "arguments": []}
    for span, role in spans_and_roles:
        text = " ".join(tokens[span[0]: span[1]])
        enc_span = f"{span[0]}:{span[1]}"
        if role.upper().startswith("REL"):
            pos = get_pos_for_span(span, tree)
            predicate = {"span": enc_span, "text": text, "pos": pos, "role_type": role}
            frame['predicate'] = predicate
        else:
            argument = {'text': text, "span": enc_span, "arg_type": "explicit", "role_type": role}
            frame['arguments'].append(argument)
    return frame


def main(args):
    nombank_path = os.path.join(args.nombank_dir, "nombank.1.0")
    mrg_dir = os.path.join(args.penntreebank_dir, "PARSED", "MRG")
    corpus = BracketParseCorpusReader(mrg_dir, WSJ_FILE_PATTERN)
    doc_to_frames = defaultdict(list)
    doc_to_sent = {}

    with open(nombank_path) as file_in:
        curr_doc_id, curr_sent_id = None, None
        for line in tqdm(file_in, desc="Parsing NomBank", total=114574):
            if 'ARG' not in line:
                continue
            splits = line.split()
            file_id, sent_id, pred_idx, pred_lemma, pred_sense = splits[:5]
            doc_id = os.path.splitext(os.path.basename(file_id))[0]
            sent_id, pred_idx = int(sent_id), int(pred_idx)
            # The frames are in sorted document order in nombank.
            if curr_doc_id != doc_id:
                tree: Tree = corpus.parsed_sents(fileids=[file_id])[sent_id]
                tokens: List[str] = corpus.sents(fileids=[file_id])[sent_id]
                tagged_doc_sents = list(corpus.tagged_sents(fileids=[file_id]))
            elif curr_sent_id != sent_id:
                tokens: List[str] = corpus.sents(fileids=[file_id])[sent_id]
                tree: Tree = corpus.parsed_sents(fileids=[file_id])[sent_id]
            curr_doc_id = doc_id
            curr_sent_id = sent_id
            members = splits[5:]
            frame = parse_nominal_frame(members, tree, tokens)
            frame['predicate']['predicate_lemma'] = pred_lemma
            frame['predicate']['frame_id'] = pred_sense
            frame['sent_id'] = sent_id
            new_frame = clean_frame(frame, tagged_doc_sents)
            doc_to_frames[(doc_id, sent_id)].append(new_frame)
            doc_to_sent[(doc_id, sent_id)] = new_frame['text']
            del new_frame['text']

    all_sents = []
    for (doc_id, sent_id), frames in doc_to_frames.items():
        text = doc_to_sent[(doc_id, sent_id)]
        sent = {"doc_id": doc_id, "sent_id": sent_id, "frames": frames, "text": text}
        all_sents.append(sent)
    with jsonlines.open(args.out_path, "w") as fout:
        fout.write_all(all_sents)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--nombank_dir", help="C:\\dev\\nombank.1.0")
    ap.add_argument("--penntreebank_dir", help="C:\\dev\\ptb\TREEBANK_3")
    ap.add_argument("--out_path", default="./nombank/nombank.jsonl")
    DEBUG_WH_SINGLE_TOKENS = []
    main(ap.parse_args())

