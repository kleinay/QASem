from typing import Tuple, List
import numpy as np
import torch


def get_token_offsets(tokens: List[str]) -> List[Tuple[int, int]]:
    token_offsets = []
    char_idx = 0
    for tok in tokens:
        n_chars = len(tok)
        token_offsets.append((char_idx, char_idx + n_chars))
        char_idx += n_chars + 1  # +1 for the SINGLE space separator (assume all inputs come pretokenized)
    # Why did I add padding to the original tokens? Don't remember...
    # diff = max(0, max_length - len(token_offsets))
    # token_offsets.extend((-1, -1) for _ in range(diff))
    return token_offsets


def _has_overlap(a_offset: Tuple[int, int], b_offset: Tuple[int, int]):
    max_starts = max(a_offset[0], b_offset[0])
    min_ends = min(a_offset[1], b_offset[1])
    diff = min_ends - max_starts
    return diff > 0


def _is_empty(offset: Tuple[int, int]):
    return offset[0] == offset[1]


def find_offset_index(an_offset: Tuple[int, int], offsets: List[Tuple[int, int]]) -> int:
    if _is_empty(an_offset):
        return -1
    for tok_idx, tok_offset in enumerate(offsets):
        # reached the padding tokens
        if _is_empty(tok_offset):
            break
        if _has_overlap(an_offset, tok_offset):
            return tok_idx
    return -1


# def find_subword_index(token_idx, token_offsets, subw_offsets, initial_shift):
#     if token_idx == len(token_offsets):
#         # Edge case when the token index points after the sentence ends
#         # Happens for exclusive end tokens
#         last_valid_subword_idx = find_text_end_index(subw_offsets)
#         # to compensate for the question subwords
#         last_valid_subword_idx += initial_shift
#         return last_valid_subword_idx + 1
#
#     token_offset = token_offsets[token_idx]
#     subw_idx = find_offset_index(token_offset, subw_offsets)
#     if subw_idx == -1:
#         # No overlapping offset was found
#         return -1
#     subw_idx += initial_shift  # to compensate for the question subwords
#     return subw_idx


def find_text_start_end_indices(all_offsets: List[Tuple[int, int]]):
    start_indices = []
    end_indices = []
    for idx, curr_offset in enumerate(all_offsets[:-1]):
        next_offset = all_offsets[idx+1]
        is_next_empty = _is_empty(next_offset)
        is_curr_empty = _is_empty(curr_offset)
        if is_next_empty and not is_curr_empty:
            end_indices.append(idx)
        if not is_next_empty and is_curr_empty:
            start_indices.append(idx + 1)
    return start_indices[-1], end_indices[-1]


# def find_text_start_index(all_offsets: List[Tuple[int, int]]):
#     # The sub-word offsets restart from zero with the second text in the Q-A pair.
#     # We have to find this restart point and supply the offsets from that point
#     # to the decoding script
#     restart_indices = [idx for idx, offset in enumerate(all_offsets)
#                        if not _is_empty(offset) and _is_empty(all_offsets[idx - 1])]
#     text_start = restart_indices[-1]
#     return text_start
#
#
# def find_text_end_index(all_offsets: List[Tuple[int, int]]):
#     text_end_indices = [idx for idx, offset in enumerate(all_offsets[:-1])
#                         if not _is_empty(offset) and _is_empty(all_offsets[idx + 1])]
#     # edge case for the last token:
#     final_tok_idx = len(all_offsets) - 1
#     if not _is_empty(all_offsets[-1]):
#         text_end_indices.append(final_tok_idx)
#
#     return text_end_indices[-1]


def _find_dynamic_max_length(questions, texts):
    len_questions = [len(q.split()) for q in questions]
    len_texts = [len(t.split()) for t in texts]
    max_len_tokens = max(len_questions) + max(len_texts)
    # 1.6 was measured from SQuAD dev set as the upper bound on 90% of the data.
    # This is the ratio between number of subwords in the QA pair to the number
    # of tokens, in a BERT sub-word tokenizer.
    assumed_len_subwords = max_len_tokens * 1.6
    # THIS IS STUPID, but code readability is really important
    if assumed_len_subwords < 128:
        return 128
    elif assumed_len_subwords < 192:
        return 192
    elif assumed_len_subwords < 256:
        return 256
    elif assumed_len_subwords < 384:
        return 384
    else:
        return 512


def _shift_index(idx: int, pred_idx: int, is_span_start):
    new_idx = idx
    if idx > pred_idx:
        new_idx += 2
    elif idx == pred_idx and is_span_start:
        new_idx += 1
    return new_idx


def shift_back_index(idx: int, surrounded_pred_idx: int):
    new_idx = idx
    # This method is coupled to the _shift_index method
    # Assumes that span start indices can only point
    # directly at the surrounded predicate token
    # and the span end indices can only point at the first * token.
    # so shifting back is a bit easier.
    if new_idx > surrounded_pred_idx + 1:
        new_idx -= 2
    elif (surrounded_pred_idx - 1) <= new_idx <= (surrounded_pred_idx + 1):
        # Whatever type of index you are, if you point within * pred *
        # then move back to the original predicate position.
        new_idx = surrounded_pred_idx - 1
    return new_idx


def batch_remove_predicate_indicators(samples, spans):
    new_samples, new_spans = [], []
    for sample, span in zip(samples, spans):
        new_sample, new_span = remove_predicate_indicators(sample, span)
        new_samples.append(new_sample)
        new_spans.append(new_span)
    return new_samples, new_spans


def remove_predicate_indicators(sample, predicted_span):
    if 'predicate_span' not in sample:
        return sample, predicted_span
    predicate_span = sample['predicate_span']
    if isinstance(predicate_span, str):
        pred_idx, _ = parse_span(predicate_span)
    else:
        pred_idx = predicate_span[0]
    tokens = sample['text'].split()
    # Edge case: the predicate is not surrounded with * pred *
    # if its still located at the sentence bounds
    if pred_idx == 0 or pred_idx == (len(tokens)-1):
        return sample, predicted_span
    if tokens[pred_idx - 1] != "*" or tokens[pred_idx+1] != "*":
        return sample, predicted_span
    orig_tokens = tokens[:(pred_idx - 1)] + [tokens[pred_idx]] + tokens[pred_idx+2:]
    start, end = predicted_span
    new_start = shift_back_index(start, pred_idx)
    new_end = shift_back_index(end, pred_idx)
    new_predicted_span = new_start, new_end
    answer_spans = sample.get("gold_answer_spans", [])
    new_answer_spans, new_answers = [], []
    for ans_start, ans_end in answer_spans:
        new_answer_span = shift_back_index(ans_start, pred_idx), shift_back_index(ans_end, pred_idx)
        new_answer = ' '.join(orig_tokens[new_answer_span[0]: new_answer_span[1]])
        new_answer_spans.append(new_answer_span)
        new_answers.append(new_answer)
    new_pred_idx = shift_back_index(pred_idx, pred_idx)
    new_sample = dict(sample)
    new_sample['text'] = ' '.join(orig_tokens)
    new_sample['gold_answers'] = new_answers
    new_sample['gold_answer_spans'] = new_answer_spans
    new_sample['predicate_span'] = f"{new_pred_idx}:{new_pred_idx+1}"
    return new_sample, new_predicted_span


def surround_predicate_with_indicators(sample):
    if 'predicate_span' not in sample:
        return sample
    pred_span = sample['predicate_span']
    if isinstance(pred_span, str):
        pred_idx, _ = parse_span(pred_span)
    else:
        pred_idx = pred_span[0]

    tokens = sample['text'].split()
    new_tokens = tokens[:pred_idx] + ["*", tokens[pred_idx], "*"] + tokens[(pred_idx + 1):]
    new_text = " ".join(new_tokens)
    answer_spans = sample.get("gold_answer_spans", [])
    new_answer_spans, new_answers = [], []
    for start, end in answer_spans:
        new_start = _shift_index(start, pred_idx, is_span_start=True)
        new_end = _shift_index(end, pred_idx, is_span_start=False)
        new_answer_spans.append((new_start, new_end))
        new_answer = ' '.join(new_tokens[new_start: new_end])
        new_answers.append(new_answer)

    # If we have candidates
    search_space = sample.get('search_space', [])
    new_search_space = []
    for cand_span in search_space:
        cand_start, cand_end = parse_span(cand_span)
        new_start = _shift_index(cand_start, pred_idx, is_span_start=True)
        new_end = _shift_index(cand_end, pred_idx, is_span_start=False)
        new_search_space.append((new_start, new_end))

    new_prd_idx = _shift_index(pred_idx, pred_idx, is_span_start=True)
    new_sample = dict(sample)
    new_sample['text'] = new_text
    new_sample['gold_answers'] = new_answers
    new_sample['gold_answer_spans'] = new_answer_spans
    new_sample['search_space'] = new_search_space
    new_sample['predicate_idx'] = new_prd_idx
    new_sample['predicate_span'] = f"{new_prd_idx}:{new_prd_idx + 1}"
    return new_sample


def get_token_to_subword_mapping(token_offsets, subword_offsets):
    tok2sub = []
    text_start_subw, text_end_subw = find_text_start_end_indices(subword_offsets)
    subw_idx = text_start_subw
    is_truncated = False
    for token_offset in token_offsets:
        # check if you have exhausted all sub-words
        # while still having tokens to account for.
        if subw_idx > text_end_subw:
            is_truncated = True
            tok2sub.append(-1)
            continue
        # for some edge examples, BERT dictionary doesn't have a literal or character
        # to represent the text. For example:
        # 'أحمد البشير‎ ‎ ;'
        # and just skips their subword completely.
        if not _has_overlap(token_offset, subword_offsets[subw_idx]):
            # continue to the next token, maybe it will match the current subword.
            tok2sub.append(-1)
            continue
        tok2sub.append(subw_idx)
        # loop until subw_idx is after the current token
        while subw_idx <= text_end_subw:
            should_stop = not _has_overlap(subword_offsets[subw_idx], token_offset)
            if should_stop:
                break
            # subw_idx points to either an offset of the next token
            # or to the final [SEP] sub-word.
            subw_idx += 1

    assert subw_idx == (text_end_subw + 1)
    # map token_idx n_tokens (the after last) to either
    # the last [SEP] subword or -1 if were truncated beforehand.
    # token n_tokens is used to translate exclusive span_end indices
    # that end the sentence.
    if is_truncated:
        tok2sub.append(-1)
    else:
        tok2sub.append(text_end_subw+1)
    return tok2sub


def translate_token_to_subword(span: Tuple[int, int], tok2sub: List[int]):
    start, end = span
    start_subw_idx = tok2sub[start]
    # End indices are exclusive, we want to take the last inclusive sub-word
    # (this might be a ##xyz continuation of the ending token).
    # If we have searched directly for end_token_idx - 1,
    # we had to continue our search for the last subword that covers
    # this token
    # Now we just shift the end sub-word back once.
    end_subw_idx = tok2sub[end]
    end_subw_idx -= 1
    # Edge case, when the answer is beyond our scope, point to the CLS token.
    if start_subw_idx < 0 or end_subw_idx < 0:
        start_subw_idx, end_subw_idx = 0, 0
    return start_subw_idx, end_subw_idx


def translate_sample(sample, encoding):
    token_offsets = get_token_offsets(sample['text'].split())
    tok2sub = get_token_to_subword_mapping(token_offsets, encoding.offsets)
    answer_span = sample['answer_start'], sample['answer_end']
    answ_subw_span = translate_token_to_subword(answer_span, tok2sub)
    trans_search_space = [translate_token_to_subword(arg['span'], tok2sub)
                          for arg in sample.get('search_space', [])]
    new_sample = dict(sample)
    new_sample['answer_start'] = answ_subw_span[0]
    new_sample['answer_end'] = answ_subw_span[1]
    new_sample['search_space'] = trans_search_space
    return new_sample


def batch_translate_token_subword_positions(samples, batch, max_length):
    keys = set(samples[0].keys())
    if 'answer_start' not in keys or 'answer_end' not in keys:
        return samples, batch, 0

    subw_starts, subw_ends = [], []
    n_answer_out_of_scope = 0
    trans_samples = []
    for sample, encoding in zip(samples, batch.encodings):
        translated_sample = translate_sample(sample, encoding)
        trans_samples.append(translated_sample)
        subw_start = translated_sample['answer_start']
        subw_end = translated_sample['answer_end']
        is_ok = subw_start > 0 and subw_end > 0
        subw_starts.append(subw_start)
        subw_ends.append(subw_end)
        if not is_ok:
            n_answer_out_of_scope += 1

    subw_starts = torch.tensor(subw_starts, dtype=torch.int64).reshape(-1, 1)
    subw_ends = torch.tensor(subw_ends, dtype=torch.int64).reshape(-1, 1)
    batch['start_positions'] = subw_starts
    batch['end_positions'] = subw_ends
    return trans_samples, batch, n_answer_out_of_scope


def pick_random_answer(sample):
    # select each time (dynamically)
    # a random gold answer from the available set of answers
    answer_spans = sample.get('gold_answer_spans', [])
    if not answer_spans:
        return sample

    n_answers = len(answer_spans)
    if n_answers > 1:
        selected_idx = np.random.randint(n_answers)
    else:
        selected_idx = 0
    start, end = answer_spans[selected_idx]
    sample['answer_start'] = start
    sample['answer_end'] = end
    return sample


class DummyModelForQuestionAnswering(torch.nn.Module):
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None):
        batch_size, max_length = input_ids.shape
        start_scores = torch.rand(batch_size, max_length)
        end_scores = torch.rand(batch_size, max_length)
        loss = torch.rand(1)
        return loss, start_scores, end_scores


def parse_span(span: str):
    start, end = span.split(":")
    return int(start), int(end)
