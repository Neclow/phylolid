"""Unit tests for the language-ID metrics in src/lid/metrics.py."""

import math

import numpy as np
import pytest
import torch

from ete4 import Tree
from sklearn.metrics import f1_score
from torch.nn.functional import one_hot
from torchmetrics import MetricCollection

from src._config import NON_LANGUAGE_GLOTTOCODE
from src.lid.metrics import (
    KnownAccuracy,
    KnownHRSeverity,
    KnownIHRSeverity,
    KnownLCAF1Severity,
    KnownMacroF1,
    KnownSeverity,
    UnseenHRSeverity,
    UnseenIHRSeverity,
    UnseenLCAF1Severity,
    UnseenPathSeverity,
    UnseenSameFamily,
    UnseenSeverity,
    hierarchical_recall,
    information_recall,
    lca_f1,
    path_distance,
    same_family,
)

# 4 known classes (3 never occurs) and one unseen language (4), whose sentences are predicted as 0 and 2
TARGET = torch.tensor([0, 0, 1, 1, 2, 4, 4])
TOP1 = torch.tensor([0, 1, 1, 1, 2, 0, 2])


@pytest.mark.parametrize(
    "preds", [one_hot(TOP1, 4).float(), TOP1], ids=["scores", "top1"]
)
def test_known_accuracy_skips_unseen_languages(preds):
    # 4 of the 5 known-language sentences are right
    assert KnownAccuracy(4)(preds, TARGET).item() == pytest.approx(4 / 5)


@pytest.mark.parametrize(
    "preds", [one_hot(TOP1, 4).float(), TOP1], ids=["scores", "top1"]
)
def test_known_macro_f1_counts_unseen_as_false_positives(preds):
    # F1 = 2tp / (2tp + fp + fn): class 0 (tp 1, fp 1, fn 1), class 1 (tp 2, fp 1), class 2 (tp 1, fp 1).
    # Class 3 has no test sentence and the unseen language is not averaged.
    expected = (2 / 4 + 4 / 5 + 2 / 3) / 3
    assert KnownMacroF1(4)(preds, TARGET).item() == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(10))
def test_metric_collection_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    # C known classes (the classifier's outputs), then U unseen languages
    C, U, n = (
        int(rng.integers(5, 40)),
        int(rng.integers(1, 10)),
        int(rng.integers(50, 500)),
    )
    target = torch.from_numpy(rng.integers(0, C + U, n))
    scores = torch.randn(n, C, generator=torch.Generator().manual_seed(seed))
    # Right about half the time on known-language sentences
    right = (target < C) & torch.from_numpy(rng.random(n) < 0.5)
    scores[right, target[right]] += 10

    # As in a LightningModule: a cloned collection, called once per batch
    metric = MetricCollection([KnownAccuracy(C), KnownMacroF1(C)]).clone()
    for batch in torch.arange(n).tensor_split(4):
        metric(scores[batch], target[batch])
    result = metric.compute()

    top1 = scores.argmax(-1)
    known = target < C
    assert result["KnownAccuracy"].item() == pytest.approx(
        (top1[known] == target[known]).float().mean().item()
    )
    expected_f1 = f1_score(
        target.numpy(),
        top1.numpy(),
        labels=target[known].unique().numpy(),
        average="macro",
    )
    assert result["KnownMacroF1"].item() == pytest.approx(expected_f1)


# Two families, F and K. Classes 0..2 are a, c, d; b (3) and e (4) are unseen languages
LINEAGES = {
    "a": ("F", "G1", "a"),
    "c": ("F", "G2", "c"),
    "d": ("K", "d"),
    "b": ("F", "G1", "b"),
    "e": ("K", "e"),
}
CLASSES = ["a", "c", "d"]


def test_hierarchical_recall():
    recall = hierarchical_recall(
        list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES]
    )
    # b shares F and G1 with a (2 of 3), only F with c, nothing with d; e shares K with d (1 of 2)
    assert recall[3].tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    assert recall[4].tolist() == pytest.approx([0, 0, 1 / 2])
    assert recall[0, 0].item() == 1


def test_hierarchical_recall_dialect():
    # A dialect x of a: predicting a recovers 3 of its 4 lineage entries
    recall = hierarchical_recall([("F", "G1", "a", "x")], [LINEAGES["a"]])
    assert recall[0].tolist() == pytest.approx([3 / 4])


@pytest.mark.parametrize("measure", [hierarchical_recall, information_recall, lca_f1])
def test_non_language_class_is_its_own_family(measure):
    # As a class and as a true label: 1 when predicted for itself, 0 for or against any language
    lineages = [LINEAGES["a"], LINEAGES["c"], (NON_LANGUAGE_GLOTTOCODE,)]
    table = measure(lineages, lineages)
    assert table[2].tolist() == pytest.approx([0, 0, 1])
    assert table[:, 2].tolist() == pytest.approx([0, 0, 1])


def test_information_recall():
    recall = information_recall(list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES])
    # Over the 5 true languages, I(group) = log2 5 - log2 (languages below it): G1 has 2 (a, b), F 3, K 2.
    # A language's own information is log2 5
    i = math.log2(5)
    assert recall[3].tolist() == pytest.approx([(i - 1) / i, (i - math.log2(3)) / i, 0])
    assert recall[4].tolist() == pytest.approx([0, 0, (i - 1) / i])
    assert recall[0, 0].item() == 1


def test_lca_f1():
    f1 = lca_f1(list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES])
    # 2 / (p + 2): b to a is 2 levels (b, G1, a), b to c 4, e to d 2; nothing across families
    assert f1[3].tolist() == pytest.approx([2 / 4, 2 / 6, 0])
    assert f1[4].tolist() == pytest.approx([0, 0, 2 / 4])
    assert f1[0, 0].item() == 1


def test_lca_f1_ignores_depth():
    # Sister languages score the same at depth 3 and depth 6, unlike hR
    shallow = [("F", "G", "x")], [("F", "G", "y")]
    deep = [("F", "G", "H", "I", "J", "x")], [("F", "G", "H", "I", "J", "y")]
    assert lca_f1(*shallow).item() == lca_f1(*deep).item() == 2 / 4
    assert hierarchical_recall(*shallow).item() < hierarchical_recall(*deep).item()


@pytest.mark.parametrize("as_scores", [True, False], ids=["scores", "top1"])
def test_unseen_severity_averages_per_language(as_scores):
    cost = 1 - hierarchical_recall(
        list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES]
    )
    # A known sentence (ignored), three sentences of b (predicted a, a, c) and one of e (predicted d)
    target = torch.tensor([0, 3, 3, 3, 4])
    top1 = torch.tensor([1, 0, 0, 1, 2])
    preds = one_hot(top1, 3).float() if as_scores else top1
    metric = MetricCollection([UnseenSeverity(cost)])
    for batch in torch.arange(5).tensor_split(2):
        metric(preds[batch], target[batch])
    # b: 1 - (2/3 + 2/3 + 1/3) / 3 = 4/9; e: 1 - 1/2. Each language counts once, whatever its number of sentences
    result = metric.compute()["UnseenSeverity"]
    assert result.item() == pytest.approx((4 / 9 + 1 / 2) / 2)


def test_known_severity_counts_right_answers_as_zero():
    cost = 1 - hierarchical_recall(
        list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES]
    )
    # a: right once, predicted c once (cost 1 - 1/3); c and d: right; the unseen sentence (b) is ignored
    target = torch.tensor([0, 0, 1, 2, 3])
    top1 = torch.tensor([0, 1, 1, 2, 0])
    metric = MetricCollection([KnownSeverity(cost)])
    metric(top1, target)
    # a: (0 + 2/3) / 2; c and d: 0
    assert metric.compute()["KnownSeverity"].item() == pytest.approx(
        (1 / 3 + 0 + 0) / 3
    )


def test_known_and_unseen_severity_together():
    # In one collection, each metric keeps its own sentences
    cost = 1 - lca_f1(list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES])
    target = torch.tensor([0, 1, 3, 4])
    top1 = torch.tensor([1, 1, 0, 2])
    metric = MetricCollection([KnownSeverity(cost), UnseenSeverity(cost)])
    metric(top1, target)
    result = metric.compute()
    # Known: a predicted c (p = 4, cost 1 - 2/6), c right. Unseen: b predicted a (p = 2), e predicted d (p = 2)
    assert result["KnownSeverity"].item() == pytest.approx((2 / 3 + 0) / 2)
    assert result["UnseenSeverity"].item() == pytest.approx((1 / 2 + 1 / 2) / 2)


@pytest.mark.parametrize(
    "known, unseen, measure",
    [
        (KnownHRSeverity, UnseenHRSeverity, hierarchical_recall),
        (KnownIHRSeverity, UnseenIHRSeverity, information_recall),
        (KnownLCAF1Severity, UnseenLCAF1Severity, lca_f1),
    ],
    ids=["hR", "ihR", "LCA-hF"],
)
def test_named_severities_build_their_cost(known, unseen, measure):
    # LINEAGES lists the 3 classes first, then the unseen languages
    lineages = list(LINEAGES.values())
    cost = 1 - measure(lineages, lineages[:3])
    target = torch.tensor([0, 0, 1, 3, 3, 4])
    top1 = torch.tensor([0, 1, 1, 0, 1, 2])
    named = MetricCollection([known(lineages, 3), unseen(lineages, 3)])
    generic = MetricCollection([KnownSeverity(cost), UnseenSeverity(cost)])
    named(top1, target)
    generic(top1, target)
    named_result, generic_result = named.compute(), generic.compute()
    assert named_result[known.__name__].item() == pytest.approx(
        generic_result["KnownSeverity"].item()
    )
    assert named_result[unseen.__name__].item() == pytest.approx(
        generic_result["UnseenSeverity"].item()
    )


# Reference trees of F and K, nodes named by glottocode. The deepest node is e, 0.6 below K: 2H = 1.2
TREES = {
    "F": Tree("((a:0.2,b:0.3)G1:0.1,c:0.5)F;", parser=1),
    "K": Tree("(d:0.4,e:0.6)K;", parser=1),
}


def test_path_distance():
    distance = path_distance(
        list(LINEAGES.values()), [LINEAGES[c] for c in CLASSES], TREES
    )
    # b to a: 0.3 + 0.2; b to c: 0.3 + 0.1 + 0.5; e to d: 0.6 + 0.4; across families: 2H
    assert distance[3].tolist() == pytest.approx([0.5, 0.9, 1.2])
    assert distance[4].tolist() == pytest.approx([1.2, 1.2, 1.0])
    assert distance[0, 0].item() == 0


def test_path_distance_without_node_is_nan():
    # f has no node in F's tree; family U has no tree. Across families, 2H all the same
    true = [("F", "G2", "f"), ("U", "u1")]
    classes = [LINEAGES["a"], LINEAGES["d"], ("U", "u2")]
    distance = path_distance(true, classes, TREES)
    assert distance[0, 0].isnan() and distance[1, 2].isnan()
    assert distance[0, 1].item() == distance[1, 0].item() == pytest.approx(1.2)
    # A language without a node is still 0 from itself
    assert path_distance([("F", "G2", "f")], [("F", "G2", "f")], TREES).item() == 0


def test_unseen_path_severity_skips_nan():
    # Classes a, c, d; unseen b, e and f (no node in F's tree)
    lineages = list(LINEAGES.values()) + [("F", "G2", "f")]
    target = torch.tensor([5, 5, 3, 4])
    top1 = torch.tensor([0, 2, 1, 2])
    severity = UnseenPathSeverity(lineages, 3, TREES)
    metric = MetricCollection([severity])
    metric(top1, target)
    # f: a is NaN (left out), d is 2H; b: c 0.9; e: d 1.0
    assert metric.compute()["UnseenPathSeverity"].item() == pytest.approx(
        (1.2 + 0.9 + 1.0) / 3
    )
    assert severity.missing.item() == 1


def test_unseen_same_family():
    lineages = list(LINEAGES.values())
    assert same_family(lineages, lineages[:3])[3].tolist() == [1, 1, 0]
    # b predicted a, c, d (2 of 3 in F); e predicted d (in K)
    target = torch.tensor([3, 3, 3, 4])
    top1 = torch.tensor([0, 1, 2, 2])
    metric = MetricCollection([UnseenSameFamily(lineages, 3)])
    metric(top1, target)
    assert metric.compute()["UnseenSameFamily"].item() == pytest.approx((2 / 3 + 1) / 2)
