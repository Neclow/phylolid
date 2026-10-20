"""Train a fastText-style language-ID model (src/lid/model.py) on the train split, checked on the dev split."""

import os

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src._config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_N_THREADS,
    DEFAULT_RUNS_FILE,
    DEFAULT_SPLIT_DIR,
)
from src.data.datasets import FeatureDataset, collate_features
from src.data.glottolog import lineages
from src.lid.features import SPELLINGS, Features, vocab_file
from src.lid.model import LitFastText
from src.lid.tree import huffman_lineages, newick, shuffle_classes


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--dim", type=int, default=256, help="Embedding size")
    # Phylogenetic soft labels (src/lid/model.py, soft_targets); 0: cross-entropy
    parser.add_argument(
        "--soft-lambda",
        type=float,
        default=0.0,
        help="Share of the target the true language gives to its relatives",
    )
    parser.add_argument(
        "--soft-beta",
        type=float,
        default=0.5,
        help="How fast a relative's share falls with its distance in Glottolog levels",
    )
    # Losses on the embeddings against the Glottolog tree (src/lid/tree_losses.py), after phylogeny-guided-lid; 0: off
    parser.add_argument(
        "--patristic-weight",
        type=float,
        default=0.0,
        help="Weight of 1 - correlation between class-centroid distances and patristic (tree path) distances",
    )
    parser.add_argument(
        "--quartet-weight",
        type=float,
        default=0.0,
        help="Weight of the quartet loss on class centroids",
    )
    parser.add_argument(
        "--phylonn-weight",
        type=float,
        default=0.0,
        help="Weight of the phylo-nn heads predicting each sentence's Glottolog family (H0) and H3 ancestor (0: they only probe)",
    )
    parser.add_argument(
        "--tree-softmax",
        action="store_true",
        help="Output layer: choices down the tree (--tree) instead of one softmax over the classes",
    )
    parser.add_argument(
        "--tree",
        choices=["glottolog", "huffman"],
        default="glottolog",
        help="Tree of the phylogenetic methods: Glottolog's, or fastText's Huffman tree over the training sentences per class (no linguistic information)",
    )
    parser.add_argument(
        "--shuffle-tree",
        action="store_true",
        help="Control: the classes shuffled over their places in the tree (with --seed), its shape kept",
    )
    parser.add_argument(
        "--spelling",
        choices=SPELLINGS,
        default="raw",
        help="How the features read the text: as written, normalized (lowercase, no Latin or Arabic diacritics, tone or vowel marks), or both (pixi run build_vocab --spelling first)",
    )
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
    args = parser.parse_args()
    # The phylo-nn heads predict named levels (family, H3), which a Huffman tree does not have
    if args.phylonn_weight and args.tree == "huffman":
        parser.error("--phylonn-weight needs --tree glottolog")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    # wandb project and entity from .env (WANDB_PROJECT, WANDB_ENTITY; see .env.example). Connecting first records the
    # run's arguments and fails fast on a wrong project or entity, before minutes of loading the splits. A dry run
    # logs nothing
    logger = False
    if not args.dry_run:
        logger = WandbLogger(save_dir=DEFAULT_EVAL_DIR)
        logger.experiment.config.update(vars(args))

    features = Features.from_json(vocab_file(args.spelling), args.spelling)
    # Classes: the training glottocodes, in a fixed order
    classes = sorted(os.listdir(f"{DEFAULT_SPLIT_DIR}/train"))
    datasets = {
        split: FeatureDataset(split, classes, features) for split in ["train", "dev"]
    }
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.n_threads,
            collate_fn=collate_features,
            pin_memory=True,
        )
        for split in ["train", "dev"]
    }
    # The tree as the classes' lineages: Glottolog's, from the datasets' mapping, or a Huffman tree
    if args.tree == "glottolog":
        lineage = lineages(pd.read_csv(f"{DEFAULT_DATASET_DIR}/glottolog.csv"))
        class_lineages = [lineage[g] for g in classes]
    else:
        counts = np.bincount(datasets["train"].targets, minlength=len(classes))
        class_lineages = huffman_lineages(classes, counts, args.seed)
    if args.shuffle_tree:
        class_lineages = shuffle_classes(class_lineages, args.seed)
    # The tree the run trains with, in its folder next to its checkpoints (<project>/<run>)
    if logger:
        run_dir = f"{DEFAULT_EVAL_DIR}/{logger.name}/{logger.version}"
        os.makedirs(run_dir, exist_ok=True)
        with open(f"{run_dir}/tree.nwk", "w") as f:
            f.write(newick(class_lineages) + "\n")
    model = LitFastText(
        features.size,
        classes,
        args.dim,
        args.lr,
        lam=args.soft_lambda,
        beta=args.soft_beta,
        patristic_weight=args.patristic_weight,
        quartet_weight=args.quartet_weight,
        phylonn_weight=args.phylonn_weight,
        tree_softmax=args.tree_softmax,
        spelling=args.spelling,
        lineages=class_lineages,
    )

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
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="valid_loss", mode="min", save_top_k=1),
            TQDMProgressBar(),
        ],
    )
    trainer.fit(model, loaders["train"], loaders["dev"])
    # The run's row in the runs table, which evaluate.py marks once it scores the run. A run stopped with Ctrl-C exits
    # in fit, so only finished runs get one; a dry run has no run
    if logger:
        runs = pd.DataFrame(
            [{"run": logger.experiment.id, **vars(args), "evaluated": False}]
        )
        if os.path.exists(DEFAULT_RUNS_FILE):
            runs = pd.concat([pd.read_csv(DEFAULT_RUNS_FILE), runs])
        runs.to_csv(DEFAULT_RUNS_FILE, index=False)


if __name__ == "__main__":
    main()
