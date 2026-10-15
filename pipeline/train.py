"""Train a fastText-style language-ID model (src/lid/model.py) on the train split, checked on the dev split."""

import os

from argparse import ArgumentParser

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src._config import DEFAULT_EVAL_DIR, DEFAULT_N_THREADS, DEFAULT_SPLIT_DIR
from src.data.datasets import FeatureDataset, collate_features
from src.lid.features import Features
from src.lid.model import LitFastText


def parse_args():
    parser = ArgumentParser(description=__doc__)
    # Defaults: ConLID's cross-entropy baseline (papers/conlid.pdf, Table 10)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--dim", type=int, default=256, help="Embedding size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=DEFAULT_N_THREADS,
        help="CPU processes computing the text features (DataLoader workers)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="One training and one dev batch, to test the pipeline",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    features = Features.from_json(f"{DEFAULT_SPLIT_DIR}/vocab.json")
    # Classes: the training glottocodes, in a fixed order
    classes = sorted(os.listdir(f"{DEFAULT_SPLIT_DIR}/train"))
    loaders = {
        split: DataLoader(
            FeatureDataset(split, classes, features),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.n_threads,
            collate_fn=collate_features,
            pin_memory=True,
        )
        for split in ["train", "dev"]
    }
    model = LitFastText(features.size, classes, args.dim, args.lr)

    if args.device.startswith("cuda"):
        accelerator, devices = "gpu", [int(args.device.split(":")[-1])]
    else:
        accelerator, devices = "cpu", "auto"
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.n_epochs,
        # Dev scores four times per epoch: training is a single epoch
        val_check_interval=0.25,
        fast_dev_run=args.dry_run,
        # wandb project and entity from .env (WANDB_PROJECT, WANDB_ENTITY; see .env.example)
        logger=WandbLogger(save_dir=DEFAULT_EVAL_DIR),
        callbacks=[
            ModelCheckpoint(monitor="valid_loss", mode="min", save_top_k=1),
            TQDMProgressBar(),
        ],
    )
    trainer.fit(model, loaders["train"], loaders["dev"])


if __name__ == "__main__":
    main()
