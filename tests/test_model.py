"""Unit tests for the fastText-style model in src/lid/model.py."""

import torch

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from src.data.datasets import FeatureDataset, collate_features
from src.lid.features import Features
from src.lid.model import FastText, LitFastText


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fasttext_is_mean_then_linear():
    model = FastText(n_features=10, n_classes=3, dim=4)
    # Two texts: ids [1, 2, 2] and [7]
    logits = model(torch.tensor([1, 2, 2, 7]), torch.tensor([0, 3]))
    mean = torch.stack(
        [model.embedding.weight[[1, 2, 2]].mean(0), model.embedding.weight[7]]
    )
    assert torch.allclose(logits, model.linear(mean))


def test_training_runs(tmp_path):
    # Two classes, a train and a dev split: one epoch on the CPU, scored on dev
    for split in ["train", "dev"]:
        write(tmp_path / split / "abcd1234" / "abc_Latn_cbc.txt", "the cat\nthe dog\n")
        write(tmp_path / split / "efgh1234" / "efg_Latn_pbc.txt", "le chat\nle chien\n")
    classes = ["abcd1234", "efgh1234"]
    features = Features(["the", "le"])
    train, dev = [
        DataLoader(
            FeatureDataset(split, classes, features, split_dir=tmp_path),
            batch_size=2,
            collate_fn=collate_features,
        )
        for split in ["train", "dev"]
    ]
    model = LitFastText(features.size, classes, dim=4, lr=0.1)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train, dev)
    assert {"valid_loss", "valid_KnownAccuracy", "valid_KnownMacroF1"} <= set(
        trainer.callback_metrics
    )
