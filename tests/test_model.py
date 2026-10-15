"""Unit tests for the fastText-style model in src/lid/model.py."""

import torch

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from src.data.datasets import FeatureDataset, collate_features
from src.lid.features import Features
from src.lid.model import FastText, LitFastText, Predictor, soft_targets


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


def test_soft_targets():
    # Family f (a and b sisters, c further), family h, an isolate, two languages of a pseudo-family
    lineages = [
        ("f", "g", "a"),
        ("f", "g", "b"),
        ("f", "c"),
        ("h", "d"),
        ("e",),
        ("arti1236", "p"),
        ("arti1236", "q"),
    ]
    # lam 0: one-hot targets, i.e. cross-entropy
    assert torch.equal(soft_targets(lineages, 0.0, 0.5), torch.eye(7))
    targets = soft_targets(lineages, 0.1, 0.5)
    assert torch.allclose(targets.sum(dim=1), torch.ones(7))
    # a keeps 0.9; b (2 levels) gets more than c (3 levels), by exp(-beta * levels); other families nothing
    weights = torch.exp(-0.5 * torch.tensor([2.0, 3.0]))
    assert torch.allclose(
        targets[0, :3], torch.cat([torch.tensor([0.9]), 0.1 * weights / weights.sum()])
    )
    assert targets[0, 3:].sum() == 0
    # No relatives: an alone family member, an isolate, the pseudo-family's members
    for y in [3, 4, 5, 6]:
        assert torch.equal(targets[y], torch.eye(7)[y])


def test_one_hot_targets_are_cross_entropy():
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 2, 1, 2])
    loss = torch.nn.CrossEntropyLoss()
    assert torch.allclose(loss(logits, torch.eye(3)[target]), loss(logits, target))


def test_training_and_prediction(tmp_path):
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
    # Soft labels on: the two classes are sisters
    model = LitFastText(
        features.size,
        classes,
        dim=4,
        lr=0.1,
        lam=0.1,
        patristic_weight=0.1,
        quartet_weight=0.1,
        phylonn_weight=0.1,
        lineages=[("f", "abcd1234"), ("f", "efgh1234")],
    )
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
    # The checkpoint behind the zoo models' interface: same classes, probabilities of the trained model
    trainer.save_checkpoint(tmp_path / "model.ckpt")
    predictor = Predictor(str(tmp_path / "model.ckpt"), features)
    assert predictor.labels() == classes
    # The phylo-nn heads come back with their sizes: one family (H0); each class is its own group at H3
    assert [head.out_features for head in predictor.lit.phylonn_heads] == [1, 2]
    texts = ["the cat", "le chien", "x"]
    batch = collate_features([(features(text), 0) for text in texts])
    expected = model(batch["ids"], batch["offsets"]).softmax(dim=-1)
    assert torch.allclose(torch.from_numpy(predictor.predict_proba(texts)), expected)
