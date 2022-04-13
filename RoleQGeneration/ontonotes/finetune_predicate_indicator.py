import types
from argparse import ArgumentParser

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from ontonotes.predicate_detector_model import PredicateDetectionDataset, PredicateDetectionModule


def total_train_steps(args, train_set: Dataset) -> int:
    """The number of total training steps that will be run. Used for lr scheduler purposes."""
    is_default = isinstance(args.gpus, types.FunctionType)
    gpus = 0 if is_default else args.gpus
    if isinstance(gpus, str) and ',' in gpus:
        gpus = len(gpus.split(","))

    num_devices = max(1, gpus)
    effective_batch_size = args.batch_size * args.accumulate_grad_batches * num_devices
    return (len(train_set) / effective_batch_size) * args.max_epochs


def main(args):
    data_path = "./ontonotes/ontonotes.{0}.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    samples = {
        "train": PredicateDetectionDataset.load_samples(data_path.format("train"), args.predicate_pos),
        "dev": PredicateDetectionDataset.load_samples(data_path.format("dev"), args.predicate_pos)
    }

    datasets = {
        "train": PredicateDetectionDataset(samples["train"], tokenizer, max_length=args.max_length),
        "dev": PredicateDetectionDataset(samples["dev"], tokenizer, max_length=args.max_length)
    }

    train_loader = DataLoader(datasets['train'],
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=datasets['train'].collate)
    dev_loader = DataLoader(datasets['dev'],
                            batch_size=2*args.batch_size,
                            shuffle=False,
                            collate_fn=datasets['dev'].collate)

    logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir, name=args.exp_name)
    max_epochs = args.max_epochs if args.max_epochs < 1000 else 5
    n_grad_update_steps = total_train_steps(args, datasets['train'])
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)
    # instantiate pytorch-lightning module with optimization parameters
    pl_module = PredicateDetectionModule(
        args.model_name,
        n_grad_update_steps,
        args.weight_decay,
        args.learning_rate,
        args.adam_eps).load_model(model)

    model_cp = pl.callbacks.ModelCheckpoint(monitor='val_f1', mode='max')
    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_f1',
        min_delta=0.05,
        patience=3,
        verbose=True,
        mode='max'
    )

    # Useful training features:
    # val_check_interval: How often within one training epoch to check the validation set. Can specify as float or int.
    #   use (float) to check within a training epoch
    #   use (int) to check every n steps (batches)
    # limit_val_batches: How much of validation dataset to check (floats = percent, int = num_batches)
    #                    upon train epoch end.
    # overfit_batches:  Uses this much data of the training set. If nonzero, will use the same training set
    # for validation and testing. If the training dataloaders have shuffle=True,
    # Lightning will automatically disable it. Useful for quickly debugging or trying to overfit on purpose.
    trainer = pl.Trainer(gpus=args.gpus,
                         checkpoint_callback=model_cp,
                         callbacks=[early_stop],
                         distributed_backend=args.distributed_backend,
                         logger=logger,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         val_check_interval=args.val_check_interval,
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         max_epochs=max_epochs,
                         fast_dev_run=args.fast_dev_run,
                         weights_save_path=args.save_dir)
    trainer.fit(pl_module, train_loader, dev_loader)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--predicate_pos", default="n", options=["n", "v"])
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--save_dir", default="./experiments")
    ap.add_argument("--exp_name", default="nom_pred")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    ap.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    ap.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    ap.add_argument("--max_length", default=128, type=int)
    ap = pl.Trainer.add_argparse_args(ap)
    main(ap.parse_args())
