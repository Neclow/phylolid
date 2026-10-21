"""Unit tests for the fastText-style model in src/lid/model.py."""

import pytest
import torch

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from src.data.datasets import FeatureDataset, collate_features
from src.lid.features import Features
from src.lid.model import FastText, LitFastText, Predictor, TreeSoftmax, soft_targets
from src.lid.tree import tree_softmax_layout


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


def test_hxe_loss():
    # Family f (a and b sisters below g, c further), a language l above its dialect d, an isolate e
    lineages = [
        ("f", "g", "a"),
        ("f", "g", "b"),
        ("f", "c"),
        ("f", "l"),
        ("f", "l", "d"),
        ("e",),
    ]
    logits = torch.randn(3, 6)
    target = torch.tensor([0, 3, 5])
    hxe = lambda alpha: LitFastText(
        10, list("abcdef"), 4, 0.1, hxe_alpha=alpha, lineages=lineages
    ).hxe_loss(logits, target)
    # alpha 0: the conditionals' logs add up to cross-entropy
    assert torch.allclose(hxe(0.0), torch.nn.functional.cross_entropy(logits, target))
    # a: f at depth 1, g | f at depth 2, a | g at depth 3; l: f, then l | f, then l itself | l (its dialect below it) at
    # depth 3; the isolate e: e at depth 1
    p = logits.softmax(dim=-1)
    w = lambda depth: torch.exp(torch.tensor(-0.5 * depth))
    log = torch.log
    expected = -torch.stack(
        [
            w(1) * log(p[0, :5].sum())
            + w(2) * log(p[0, :2].sum() / p[0, :5].sum())
            + w(3) * log(p[0, 0] / p[0, :2].sum()),
            w(1) * log(p[1, :5].sum())
            + w(2) * log(p[1, 3:5].sum() / p[1, :5].sum())
            + w(3) * log(p[1, 3] / p[1, 3:5].sum()),
            w(1) * log(p[2, 5]),
        ]
    ).mean()
    assert torch.allclose(hxe(0.5), expected, atol=1e-6)


def test_tree_softmax():
    # Family f (a and b sisters, c further), an isolate e
    layout, groups = tree_softmax_layout(
        [("f", "g", "a"), ("f", "g", "b"), ("f", "c"), ("e",)]
    )
    model = TreeSoftmax(4, layout, groups)
    embeddings = torch.randn(3, 4)
    log_probs = model(embeddings)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(3))
    # a: f among the root's children, g among f's, a among g's (the first unit of each group)
    scores = model.linear(embeddings)
    choices = [scores[:, groups == group].softmax(dim=1)[:, 0] for group in range(3)]
    assert torch.allclose(log_probs[:, 0].exp(), choices[0] * choices[1] * choices[2])


@pytest.mark.parametrize(
    "tree_softmax, gradnorm, hxe_alpha",
    [(False, False, 0.0), (True, False, 0.0), (False, True, 0.0), (False, False, 0.5)],
)
def test_training_and_prediction(tmp_path, tree_softmax, gradnorm, hxe_alpha):
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
    # Soft labels on, unless hierarchical cross-entropy: the two classes are sisters
    model = LitFastText(
        features.size,
        classes,
        dim=4,
        lr=0.1,
        lam=0.0 if hxe_alpha else 0.1,
        patristic_weight=0.1,
        quartet_weight=0.1,
        phylonn_weight=0.1,
        tree_softmax=tree_softmax,
        gradnorm=gradnorm,
        hxe_alpha=hxe_alpha,
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
    # GradNorm moved the four losses' weights from 1, still summing to 4
    if gradnorm:
        assert not torch.allclose(model.loss_weights, torch.ones(4))
        assert model.loss_weights.sum().item() == pytest.approx(4)
    # One step per batch: SparseAdam for the embedding table, AdamW for the rest, both down to a learning rate of 0 (and
    # Adam for GradNorm's weights)
    assert trainer.global_step == len(train)
    sparse, dense, *weights = trainer.optimizers
    assert isinstance(sparse, torch.optim.SparseAdam)
    assert isinstance(dense, torch.optim.AdamW)
    assert sparse.param_groups[0]["params"] == [model.model.embedding.weight]
    assert len(weights) == gradnorm
    assert [o.param_groups[0]["lr"] for o in [sparse, dense]] == pytest.approx([0, 0])
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


def test_gradnorm_raises_a_small_loss():
    lineages = [("f", "abcd1234"), ("f", "efgh1234")]
    model = LitFastText(
        10,
        ["abcd1234", "efgh1234"],
        4,
        0.1,
        patristic_weight=1.0,
        gradnorm=True,
        lineages=lineages,
    )
    # Two texts: ids [1, 2] and [7]
    embeddings = model.model.embedding(torch.tensor([1, 2, 7]), torch.tensor([0, 2]))
    # An extra loss whose gradient is far smaller than cross-entropy's: GradNorm raises its weight, lowers
    # cross-entropy's
    loss = model.gradnorm_loss([(10 * embeddings).sum(), (0.01 * embeddings).sum()])
    loss.backward()
    grad = model.loss_weights.grad
    assert grad is not None and grad[0] > 0 and grad[1] < 0
