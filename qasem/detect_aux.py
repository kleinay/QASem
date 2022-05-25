have_forms = {
    "have",
    "having",
    "'ve",
    "has",
    "had",
    "'d"
  }

do_forms = {
    "do",
    "does",
    "doing",
    "did",
    "done"
}

be_forms = {
    "be",
    "being",
    "been",
    "am",
    "'m",
    "is",
    "'s",
    "ai",
    "are",
    "'re",
    "was",
    "were"
}

negation_word = {
                "no",
                "not",
                "n't"
                }


def is_auxiliary_or_be(token,token_id, pos) -> bool:

    """
    pos: [(word, pos_word),....]
    return True if token is auxiliary or if token is be
    """
    if token.lower() in be_forms:
        return True

    elif token.lower() in have_forms:
        if pos[token_id + 1][0] in negation_word:
            return False
        i = 1
        while pos[token_id + i][0] in negation_word or pos[token_id + i][1] == 'ADV':
            i += 1
        if pos[token_id + i][1] == 'VBN':
            return False
        return True

    elif token.lower() in do_forms:
        
