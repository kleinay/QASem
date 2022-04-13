import os
from typing import Dict, Any, Tuple, List
import numpy as np
import pytorch_lightning as pl
from tokenizers import Encoding
from transformers import BatchEncoding, get_linear_schedule_with_warmup
import torch
from torch.optim.adamw import AdamW
import logging

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase

from RoleQGeneration.qa_models.qa_utils import find_offset_index, get_token_offsets, \
    find_text_start_end_indices

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.DEBUG)

NO_DECAY = ['bias', 'LayerNorm.weight']
BERT_SEP_TOKEN_ID = 102


def is_in_no_decay(param_name: str):
    for nd in NO_DECAY:
        if nd in param_name:
            return True
    return False


def batch_iou(gold_starts: torch.Tensor, gold_ends: torch.Tensor, pred_starts: torch.Tensor, pred_ends: torch.Tensor):
    # make columns, make end indices exclusive for length computations.
    gold_starts = gold_starts.view(-1, 1)
    gold_ends = gold_ends.view(-1, 1) + 1
    pred_starts = pred_starts.view(-1, 1)
    pred_ends = pred_ends.view(-1, 1) + 1
    # find intersection points
    max_start = torch.cat((gold_starts, pred_starts), dim=-1).max(dim=-1).values
    min_end = torch.cat((gold_ends, pred_ends), dim=-1).min(dim=-1).values
    intersection_lengths = (min_end - max_start).view(-1, 1)
    is_valid_mask = (pred_ends > pred_starts).float()
    is_intersecting_mask = (intersection_lengths >= 0).float()
    # mask-out non overlapping spans, or badly predicted ones
    # mask = (pred_ends > pred_starts).float() *
    # 0 if invalid or not intersecting. actual intersection length everywhere else.
    intersection_lengths = intersection_lengths * (is_valid_mask * is_intersecting_mask)
    gold_lengths = (gold_ends - gold_starts)
    pred_lengths = (pred_ends - pred_starts)*is_valid_mask
    total_lengths = gold_lengths + pred_lengths
    union_lengths = total_lengths - intersection_lengths
    ious = intersection_lengths.float() / union_lengths.float()
    return ious


def _get_question(encoding: Encoding):
    # Find the first special token after the initial [CLS]
    sep_idx = encoding.special_tokens_mask.index(1, 1)
    prev_offset = encoding.offsets[0]
    question_tokens = encoding.tokens[1: sep_idx]
    question = ""
    for token, offset in zip(question_tokens, encoding.offsets[1:]):
        if offset[0] > prev_offset[1]:
            question += " "
            prev_offset = offset
        question += token
    return question


def get_examples(batch: BatchEncoding, pred_starts: torch.Tensor, pred_ends: torch.Tensor, sep_token_id):
    input_ids = batch['input_ids'].detach().cpu().tolist()
    pred_starts = pred_starts.detach().cpu().tolist()
    pred_ends = pred_ends.detach().cpu().tolist()
    gold_starts = batch['start_positions'].view(-1).cpu().tolist()
    gold_ends = batch['end_positions'].view(-1).cpu().tolist()
    n_samples = batch['input_ids'].shape[0]
    res = []
    for idx in range(n_samples):
        subwords = batch['subwords'][idx]
        if sep_token_id not in input_ids[idx]:
            # Probably a RoBERTa model
            continue
        sep_idx = input_ids[idx].index(sep_token_id)
        question_tokens = subwords[1:sep_idx]
        pred_span = slice(pred_starts[idx], pred_ends[idx] + 1)
        gold_span = slice(gold_starts[idx], gold_ends[idx] + 1)
        pred_answer = " ".join(subwords[pred_span])
        gold_answer = " ".join(subwords[gold_span])
        res.append({'question': " ".join(question_tokens),
                    'gold_answer': gold_answer,
                    'pred_answer': pred_answer})
    return res


def construct_search_space(start_idx, end_idx):
    # construct a distribution from all valid indices.
    # This is a bit different than itertools.combinations
    index_pairs = []
    for i in range(start_idx, end_idx):
        for j in range(i, end_idx):
            index_pairs.append((i, j))
    index_pairs = torch.tensor(index_pairs, dtype=torch.int64)
    return index_pairs




class QuestionAnswerModule(pl.LightningModule):
    MODEL_FIELDS = {'input_ids', 'attention_mask', 'token_type_ids',
                    'start_positions', 'end_positions',
                    'predicate_idx'}

    DECODE_MODE_ARGMAX_VALID_SPANS = 'argmax_from_valid_spans'
    DECODE_MODE_ARGMAX_GREEDY = 'greedy_argmax'
    DECODE_MODES = [DECODE_MODE_ARGMAX_VALID_SPANS, DECODE_MODE_ARGMAX_GREEDY]

    def __init__(self, qa_model: torch.nn.Module,
                 learning_rate=None,
                 adam_eps=None,
                 weight_decay=None,
                 n_grad_update_steps=1,
                 decode_mode=DECODE_MODE_ARGMAX_VALID_SPANS,
                 sep_token_id=BERT_SEP_TOKEN_ID):

        super().__init__()
        self.qa_model = qa_model
        # Take one of the first validation batches and log its full outputs
        self.lucky_index = -1
        self.text_logger: SimpleTextLogger = None  # will be initialized in setup()
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.n_grad_update_steps = n_grad_update_steps
        self.sep_token_id = sep_token_id
        if decode_mode == self.DECODE_MODE_ARGMAX_GREEDY:
            decode_fn = self.decode_span_greedy
        elif decode_mode == self.DECODE_MODE_ARGMAX_VALID_SPANS:
            decode_fn = self.decode_span_non_greedy
        else:
            raise ValueError(f"decode_mode must be one of:"
                             f" {QuestionAnswerModule.DECODE_MODES},"
                             f" given: {decode_mode}")
        self.decode_fn = decode_fn
        self.decode_mode = decode_mode

    def setup(self, stage: str):
        # Called at the beginning of fit and test.
        # This is a good hook when you need to build models dynamically or adjust something about them.
        #  This hook is called on every process when using DDP.
        #  Args:
        #   stage: either 'fit' or 'test'
        if self.trainer.global_rank == 0:
            self.text_logger = SimpleTextLogger.from_logger(self.logger)

    def on_validation_epoch_start(self) -> None:
        if self.trainer.running_sanity_check:
            return

        n_val_batches = len(self.trainer.val_dataloaders[0])
        max_batches = self.trainer.num_val_batches[0]
        n_val_batches = min(n_val_batches, max_batches)
        self.lucky_index = np.random.randint(n_val_batches)

    def forward(self, batch: BatchEncoding):
        outputs = self.qa_model(**batch, return_dict=False)
        return outputs

    def training_step(self, batch, batch_idx):
        req_batch = self.to_required_inputs(batch)
        loss, start_logits, end_logits = self(req_batch)

        # self.log('train_loss', loss, on_epoch=True)
        # log both every step and every epoch
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        req_batch = self.to_required_inputs(batch)
        loss, start_logits, end_logits = self(req_batch)
        # Simple, top-1 decoding of the answer.
        pred_starts = start_logits.argmax(dim=-1).view(-1)
        pred_ends = end_logits.argmax(dim=-1).view(-1)
        gold_starts = batch['start_positions']
        gold_ends = batch['end_positions']
        ious = batch_iou(gold_starts, gold_ends, pred_starts, pred_ends)
        # averaged over the current batch samples
        avg_batch_iou = ious.mean()
        self.__log_examples(batch, batch_idx, pred_starts, pred_ends)

        self.log("val_iou", avg_batch_iou, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.trainer.running_sanity_check:
            return
        score = self.trainer.callback_metrics['val_iou']
        self.text_logger.log_object({"val_iou": score}, self.trainer.global_step)

    def get_lr_scheduler(self, optimizer, n_total_steps: int):
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=n_total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        # OK - let's see what we can do about the scheduler.
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.qa_model.named_parameters()
                       if not is_in_no_decay(n)],
            'weight_decay': self.weight_decay
        }, {
            'params': [p for n, p in self.qa_model.named_parameters()
                       if is_in_no_decay(n)],
            'weight_decay': 0.0,
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_eps)

        if self.n_grad_update_steps == 1:
            return optimizer

        scheduler = self.get_lr_scheduler(optimizer, self.n_grad_update_steps)
        return [optimizer], [scheduler]

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # nothing to save here
        if self.trainer.running_sanity_check:
            return

        # We save a checkpoint of our own, because getting the model directly
        # from PL checkpoint and loading it into a bare bones HuggingFace model
        # is not trivial, especially when we don't want to re-use the PL wrapper code.
        name = f"step_{checkpoint['global_step']}"
        save_dir = os.path.join(self.trainer.logger.log_dir, name)
        self.qa_model.save_pretrained(save_dir)

    @rank_zero_only
    def __log_examples(self, batch, batch_idx: int, pred_starts, pred_ends):
        if batch_idx == self.lucky_index or self.trainer.fast_dev_run:
            self.lucky_index = batch_idx
            examples = get_examples(batch, pred_starts, pred_ends, self.sep_token_id)
            for ex in examples:
                self.text_logger.log_object(ex, self.trainer.global_step)

    @classmethod
    def to_required_inputs(cls, batch: BatchEncoding):
        """
        Returns inputs with only the required fields that for the forward function of a module.
        The inputs may contain extra fields for decoding (subword_to_token) and will produce
        an error if passed to the model as: model.forward(**inputs)
        """
        raw_inputs = {key: batch[key] for key in cls.MODEL_FIELDS & batch.keys()}
        new_batch = BatchEncoding(data=raw_inputs)
        return new_batch

    def infer(self, samples, batch_inputs: BatchEncoding, device) -> List[Tuple[int, int]]:
        req_inputs = self.to_required_inputs(batch_inputs)
        req_inputs = req_inputs.to(device)
        outputs = self(req_inputs)
        if len(outputs) == 3:
            outputs = outputs[1:]
        # tensors of [batch_size, max_length]
        start_scores, end_scores = outputs
        spans = self.decode_scores(start_scores, end_scores, samples, batch_inputs)
        return spans

    def decode_scores(self, start_scores: torch.Tensor, end_scores: torch.Tensor, samples, batch: BatchEncoding):
        subword_offsets = [enc.offsets for enc in batch.encodings]

        start_scores = start_scores.detach().cpu()
        end_scores = end_scores.detach().cpu()
        batch_sz, max_length = start_scores.shape
        spans = []
        # sometimes, a simple for loop is just the simplest solution
        for idx in range(batch_sz):
            sample = samples[idx]
            text_tokens = sample['text'].split()
            search_space = sample.get('search_space', [])
            token_offsets = get_token_offsets(text_tokens)
            subw_offsets = subword_offsets[idx]

            span = self.decode_fn(start_scores[idx],
                                  end_scores[idx],
                                  subw_offsets,
                                  token_offsets,
                                  search_space)
            spans.append(span)
        return spans

    @staticmethod
    def decode_span_greedy(start_scores: torch.Tensor, end_scores: torch.Tensor,
                           subword_offsets: List[Tuple[int, int]],
                           token_offsets: List[Tuple[int, int]],
                           search_space) -> Tuple[int, int]:
        start_idx = start_scores.argmax(dim=-1).item()
        end_idx = end_scores.argmax(dim=-1)
        text_start_idx, _ = find_text_start_end_indices(subword_offsets)
        if start_idx == 0 and end_idx == 0:
            return 0, 0

        # The predicted answer nodes point at tokens from the question.
        if (start_idx < text_start_idx) or (end_idx < text_start_idx):
            return -1, -1

        # token offsets already relate only to the second text (the context)
        start_offset, end_offset = subword_offsets[start_idx], subword_offsets[end_idx]
        start_token = find_offset_index(start_offset, token_offsets)
        end_token = find_offset_index(end_offset, token_offsets)
        # One of the predicted sub-words comes from either
        # the special symbols or the question tokens
        is_valid = (start_token <= end_token) and (start_token != -1) and (end_token != -1)
        if not is_valid:
            return -1, -1

        return start_token, end_token + 1

    @staticmethod
    def decode_span_non_greedy(start_scores: torch.Tensor, end_scores: torch.Tensor,
                               subword_offsets: List[Tuple[int, int]],
                               token_offsets: List[Tuple[int, int]],
                               search_space: List[Tuple[int, int]]):

        if not search_space:
            text_start_idx, text_end_idx = find_text_start_end_indices(subword_offsets)
            index_pairs = construct_search_space(text_start_idx, text_end_idx + 1)
        else:
            index_pairs = torch.tensor(search_space, dtype=torch.int64)

        scores = start_scores[index_pairs[:, 0]] + end_scores[index_pairs[:, 1]]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        # now translate subword indices into token indices
        subword_start_idx, subword_end_idx = index_pairs[best_idx]
        subword_start_offset = subword_offsets[subword_start_idx.item()]
        subword_end_offset = subword_offsets[subword_end_idx.item()]
        start_token = find_offset_index(subword_start_offset, token_offsets)
        end_token = find_offset_index(subword_end_offset, token_offsets)

        # Null answers (No answer can be found in the text)
        null_score = start_scores[0] + end_scores[0]
        # We haven't calibrated a threshold, so just use any epsilon > 0.
        if null_score > best_score:
            return 0, 0
        else:
            return start_token, end_token + 1


class SimpleTextLogger:

    def __init__(self, log_dir: str):
        self.file_path = os.path.join(log_dir, 'exp.log')
        self.log_name = os.path.basename(log_dir)
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setLevel("INFO")
        fmt = logging.Formatter("%(message)s")
        file_handler.setFormatter(fmt)
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel("INFO")
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    @rank_zero_only
    def log_object(self, data: Dict[str, Any], step):
        data['step'] = step
        self.logger.info(data)

    @classmethod
    def from_logger(cls, logger: LightningLoggerBase):
        return cls(logger.log_dir)
