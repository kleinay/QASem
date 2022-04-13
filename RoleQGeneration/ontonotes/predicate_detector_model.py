from argparse import ArgumentParser
from typing import Tuple, List

import torch
from jsonlines import jsonlines
from tokenizers import Encoding
from torch.optim import AdamW
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, AutoConfig, AutoModelForTokenClassification

IGNORE_LABEL_ID = -1

def joint_len(sp1: Tuple[int, int], sp2: Tuple[int, int]):
    min_end = min(sp1[1], sp2[1])
    max_start = max(sp1[0], sp2[0])
    diff = min_end - max_start
    jlen = max(0, diff)
    return jlen


def get_token_offsets(tokens: List[str]):
    token_offsets = [(0, len(tokens[0]))]
    for tok in tokens[1:]:
        _, last_end = token_offsets[-1]
        # don't forget to add a whitespace.
        new_start = last_end + 1
        token_offsets.append((new_start, new_start + len(tok)))
    return token_offsets


def find_offset_index(offset: Tuple[int, int], offsets: List[Tuple[int, int]], start_idx: int):
    for idx, off in enumerate(offsets[start_idx:]):
        if joint_len(off, offset):
            return idx + start_idx
    return -1


def translate_token_to_subword_indices(tokens, token_indices, encoding):
    token_offsets = get_token_offsets(tokens)
    search_start_idx = 0
    token_indices = sorted(token_indices)
    subword_indices = []
    for token_idx in token_indices:
        token_offset = token_offsets[token_idx]
        subword_idx = find_offset_index(token_offset, encoding.offsets, search_start_idx)
        # predicate token was truncated
        # stop searching for the rest of the predicate tokens.
        if subword_idx < 0:
            break
        subword_indices.append(subword_idx)
        search_start_idx = subword_idx + 1
    return subword_indices


def get_predicate_indicators(samples, batch, pad_token_id) -> torch.tensor:
    n_samples, max_len = batch['input_ids'].shape[:2]
    batch_pred_indices = []
    for sample, encoding in zip(samples, batch.encodings):
        tokens = sample['text'].split()
        pred_token_indices = sample['predicate_indices']
        pred_subw_indices = translate_token_to_subword_indices(tokens,
                                                               pred_token_indices,
                                                               encoding)
        batch_pred_indices.append(pred_subw_indices)

    predicate_indicators = torch.zeros(n_samples, max_len, dtype=torch.int64)
    # where are the padding tokens? need to put -1 on every label
    # for a padding token to be ignored.
    for idx, (pred_indices, encoding) in enumerate(zip(batch_pred_indices, batch.encodings)):
        predicate_indicators[idx, pred_indices] = 1
        # If the last subword is not a padding token, then no subword is,
        # don't use the index method (will throw exception if not found)
        if encoding.ids[-1] != pad_token_id:
            continue
        pad_token_idx = encoding.ids.index(pad_token_id)
        predicate_indicators[idx, pad_token_idx:] = IGNORE_LABEL_ID
    return predicate_indicators


def parse_span(span_s: str):
    start, end = span_s.split(":")
    return int(start), int(end)


class PredicateDetectionDataset(Dataset):
    TOKENIZER_ARGS = {
        'add_special_tokens': True,
        'padding': 'max_length',
        'truncation': True,
        'max_length': 128,
        'return_tensors': 'pt'
    }

    def __init__(self, samples, tokenizer, **tokenizer_args):
        self.samples = samples
        self.tokenizer = tokenizer
        self.tok_args = dict(self.TOKENIZER_ARGS)
        self.tok_args.update(tokenizer_args)

    def collate(self, samples):
        batch, encodings = self.collate_with_encodings(samples)
        # the dict() part is because torch.scatter that is called
        # in distributed training, cannot handle properly
        # dictionaries which implement dict interface
        # but are not derived from dict.
        return dict(batch)

    def collate_with_encodings(self, samples):
        texts = [s['text'] for s in samples]
        batch = self.tokenizer(texts, **self.tok_args)
        has_predicate = 'predicate_indices' in samples[0]
        if has_predicate:
            predicate_indicators = get_predicate_indicators(samples, batch,
                                                            self.tokenizer.pad_token_id)
            batch['labels'] = predicate_indicators
        return batch, batch.encodings

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_samples(ontonotes_path: str, predicate_pos: str):
        with jsonlines.open(ontonotes_path) as file_in:
            sents = list(file_in)
        samples = []
        for sent in sents:
            frames = [frame for frame in sent['frames']
                      if frame['predicate']['pos'].lower().startswith(predicate_pos)]
            if not frames:
                continue
            predicate_spans = [frame['predicate']['span'] for frame in frames]
            predicate_indices = [parse_span(span)[0] for span in predicate_spans]
            samples.append({
                "text": sent['text'],
                "predicate_indices": predicate_indices,
                'doc_id': sent['doc_id'],
                "sent_id": sent['sent_id'],
            })
        return samples


class PredicateDetectionModule(pl.LightningModule):
    def __init__(self,
                 model_name,
                 n_grad_update_steps=0,
                 weight_decay=0.0,
                 learning_rate=3e-5,
                 adam_eps=0):
        super().__init__()
        self.save_hyperparameters()
        # replaced with either load_model or load_from_checkpoint
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_config(self.config)
        self.n_grad_update_steps = n_grad_update_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.weight_for_init_check = self._get_some_weight()
        self.is_init_weight_checked = False

    def _get_some_weight(self):
        some_weight = self.model.bert.encoder.layer[0].attention.self.query.weight
        some_weight = some_weight.detach().cpu().view(-1)[0].item()
        return some_weight

    def load_model(self, underlying_model):
        # Training script loads a pre-trained LM with
        # an uninitialized token classifier on top.
        # For inference can load from the checkpoint directly.
        self.model = underlying_model
        return self

    def on_fit_start(self):
        self.throw_if_weight_uninitialized()

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def throw_if_weight_uninitialized(self):
        if not self.is_init_weight_checked:
            some_weight = self._get_some_weight()
            if some_weight == self.weight_for_init_check:
                raise RuntimeWarning("Weights are unchanged since __init__ was called"
                                     " with random initialization from config.")
            self.is_init_weight_checked = True

    def validation_step(self, batch, batch_idx):
        # outputs.shape: batch_size x max_len x num_labels
        outputs = self(**batch)
        self.log("val_loss", outputs.loss)
        predicted_labels = outputs.logits.detach().argmax(dim=-1).view(-1)
        gold_labels = batch['labels'].detach().view(-1)
        mask = gold_labels != IGNORE_LABEL_ID
        n_tp = (gold_labels & predicted_labels & mask).sum().item()
        n_fp = (predicted_labels & ~gold_labels & mask).sum().item()
        n_fn = (~predicted_labels & gold_labels & mask).sum().item()
        n_predicted = n_tp + n_fp
        n_actual = n_tp + n_fn

        # in case where the sentence didn't have any nominals
        # and we didn't predict any token as nominal,
        # no point in calculating a zero prec/recall/f1
        # we were right after all.
        if not n_predicted and not n_actual:
            return

        precision = n_tp / n_predicted if n_predicted else 0
        recall = n_tp / n_actual if n_actual else 0
        f1 = 0.0
        if n_tp:
            f1 = (2*precision*recall)/(precision+recall)

        self.log("val_f1", f1, prog_bar=True)
        self.log("val_prec", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def configure_optimizers(self):
        def is_in_no_decay(param_name: str):
            for nd in ['bias', 'LayerNorm.weight']:
                if nd in param_name:
                    return True
            return False

        to_decay = {'params': [p for n, p in self.model.named_parameters()
                               if not is_in_no_decay(n)],
                    'weight_decay': self.weight_decay}
        not_decay = {'params': [p for n, p in self.model.named_parameters()
                                if is_in_no_decay(n)],
                     'weight_decay': 0.0}
        optimizer_grouped_parameters = [to_decay, not_decay]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_eps)

        if self.n_grad_update_steps == 1:
            return optimizer

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=self.n_grad_update_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
