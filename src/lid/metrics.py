"""Language-ID metrics over glottocode classes, in the torchmetrics API."""

import math

from collections import Counter

import torch

from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

# Classes 0..C-1 are the classifier's glottocodes, in output order. Each unseen language has its own index >= C.
# A lineage is the glottocodes from the family down to the language itself. A non-language class (und, zxx) has
# its own one-entry lineage, (NON_LANGUAGE_GLOTTOCODE,): a family of its own.
# Each function below gives an (L, C) table for L true languages and C classes. The three recall/F1 tables are 1
# when right and 0 across families.


def hierarchical_recall(true_lineages, class_lineages):
    """Hierarchical recall (Kiritchenko et al., 2006) of predicting each class for each true language."""
    # hR[y, c] = |A(y) ∩ A(c)| / |A(y)|: the share of y's lineage that c is also in
    recall = torch.zeros(len(true_lineages), len(class_lineages))
    for i, y in enumerate(true_lineages):
        for j, c in enumerate(class_lineages):
            shared = [node for node in y if node in c]
            recall[i, j] = len(shared) / len(y)
    return recall


def information_recall(true_lineages, class_lineages):
    """Information-based hierarchical recall (Valmadre, 2022) of predicting each class for each true language."""
    # A group's information is log2 n - log2 (languages below it), over the n true languages: sharing a small group
    # is worth more than sharing a large one, whatever their depths. ihR[y, c] = I(lowest shared group) / I(y)
    below = Counter(node for y in true_lineages for node in y)
    n = len(true_lineages)
    information = {
        node: math.log2(n) - math.log2(count) for node, count in below.items()
    }
    recall = torch.zeros(len(true_lineages), len(class_lineages))
    for i, y in enumerate(true_lineages):
        for j, c in enumerate(class_lineages):
            shared = [node for node in y if node in c]
            if shared:
                recall[i, j] = information[shared[-1]] / information[y[-1]]
    return recall


def lca_f1(true_lineages, class_lineages):
    """LCA-based hierarchical F1 (Kosmopoulos et al., 2015) of predicting each class for each true language."""
    # With one true and one predicted class on a tree, LCA-hF = 2 / (p + 2), p the number of levels between them:
    # it depends on the path only, not on depth
    f1 = torch.zeros(len(true_lineages), len(class_lineages))
    for i, y in enumerate(true_lineages):
        for j, c in enumerate(class_lineages):
            shared = [node for node in y if node in c]
            if shared:
                f1[i, j] = 2 / (len(y) + len(c) - 2 * len(shared) + 2)
    return f1


def path_distance(true_lineages, class_lineages, trees):
    """Path distance (sum of branch lengths) between each true language and each class in reference trees."""
    # trees: family glottocode -> its tree, nodes named by glottocode (data/trees/references/processed/).
    # Same language: 0. Another family: 2H, H the deepest node below any family root, so more than any distance
    # within a family. Same family: the path between the two nodes, NaN if either has none (e.g. no ASJP word list)
    nodes = {
        family: {node.name: node for node in tree.traverse() if node.name}
        for family, tree in trees.items()
    }
    deepest = max(
        tree.get_distance(tree, node)
        for family, tree in trees.items()
        for node in nodes[family].values()
    )
    distance = torch.full((len(true_lineages), len(class_lineages)), 2 * deepest)
    for i, y in enumerate(true_lineages):
        for j, c in enumerate(class_lineages):
            if y[-1] == c[-1]:
                distance[i, j] = 0
            elif y[0] == c[0]:
                family = nodes.get(y[0], {})
                if y[-1] in family and c[-1] in family:
                    distance[i, j] = trees[y[0]].get_distance(
                        family[y[-1]], family[c[-1]]
                    )
                else:
                    distance[i, j] = math.nan
    return distance


def same_family(true_lineages, class_lineages):
    """1 where the class is in the true language's family (the top of its lineage), 0 elsewhere."""
    return torch.tensor(
        [[float(y[0] == c[0]) for c in class_lineages] for y in true_lineages]
    )


class KnownAccuracy(MulticlassAccuracy):
    """Share of known-language sentences whose top-1 class is the true one."""

    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes, average="micro")

    def update(self, preds, target):
        # Targets >= num_classes are unseen languages
        known = target < self.num_classes
        super().update(preds[known], target[known])


class KnownMacroF1(MulticlassF1Score):
    """Mean F1 over the known languages of the test set; unseen-language sentences count as false positives."""

    def __init__(self, num_classes):
        # One extra class, num_known, collects the targets of all unseen languages (>= num_known)
        super().__init__(num_classes=num_classes + 1, average=None)
        self.num_known = num_classes

    def update(self, preds, target):
        # Scores over the known classes -> top-1 class, as torchmetrics does
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(-1)
        super().update(preds, target.clamp(max=self.num_known))

    def compute(self):
        tp, _, _, fn = self._final_state()
        # Known languages with at least one test sentence, as sklearn's f1_score(labels=...)
        return super().compute()[:-1][(tp + fn)[:-1] > 0].mean()


class Severity(Metric):
    """Mean cost of the top-1 class, per language, then over languages; subclasses choose the sentences scored."""

    full_state_update = False
    # Buffer and states, typed as torchmetrics does
    cost: torch.Tensor
    total: torch.Tensor
    count: torch.Tensor
    missing: torch.Tensor

    def __init__(self, cost):
        super().__init__()
        # cost[y, c]: how bad predicting class c is for language y, e.g. 1 - hierarchical_recall(...),
        # 1 - information_recall(...), 1 - lca_f1(...) or path_distance(...)
        self.register_buffer("cost", cost, persistent=False)
        self.add_state("total", torch.zeros(len(cost)), dist_reduce_fx="sum")
        self.add_state("count", torch.zeros(len(cost)), dist_reduce_fx="sum")
        # Scored sentences left out because their cost is NaN
        self.add_state("missing", torch.tensor(0.0), dist_reduce_fx="sum")

    def scored(self, target):
        # Mask of the sentences to score, from their targets
        raise NotImplementedError

    def update(self, preds, target):
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(-1)
        keep = self.scored(target)
        y, cost = target[keep], self.cost[target[keep], preds[keep]]
        has_cost = ~cost.isnan()
        self.missing += (~has_cost).sum()
        y, cost = y[has_cost], cost[has_cost]
        # Sum and number of costs per true language
        self.total.index_add_(0, y, cost)
        self.count.index_add_(0, y, torch.ones_like(y, dtype=self.count.dtype))

    def compute(self):
        # Languages with at least one scored sentence
        tested = self.count > 0
        return (self.total[tested] / self.count[tested]).mean()


class KnownSeverity(Severity):
    """Mean cost on known-language sentences, 0 when right: hierarchical distance@1 over all samples (Karthik et al., 2021)."""

    def scored(self, target):
        # Targets < C (the number of classes) are the classifier's own languages
        return target < self.cost.shape[1]


class UnseenSeverity(Severity):
    """Mean cost on unseen-language sentences, where every prediction is a mistake."""

    def scored(self, target):
        # Targets >= C are unseen languages
        return target >= self.cost.shape[1]


# Named severities. true_lineages: the classes first (indices 0..num_classes-1), then the unseen languages


class KnownHRSeverity(KnownSeverity):
    """1 - hR of the top-1 class on known-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(
            1 - hierarchical_recall(true_lineages, true_lineages[:num_classes])
        )


class UnseenHRSeverity(UnseenSeverity):
    """1 - hR of the top-1 class on unseen-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(
            1 - hierarchical_recall(true_lineages, true_lineages[:num_classes])
        )


class KnownIHRSeverity(KnownSeverity):
    """1 - ihR of the top-1 class on known-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(
            1 - information_recall(true_lineages, true_lineages[:num_classes])
        )


class UnseenIHRSeverity(UnseenSeverity):
    """1 - ihR of the top-1 class on unseen-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(
            1 - information_recall(true_lineages, true_lineages[:num_classes])
        )


class KnownLCAF1Severity(KnownSeverity):
    """1 - LCA-hF of the top-1 class on known-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(1 - lca_f1(true_lineages, true_lineages[:num_classes]))


class UnseenLCAF1Severity(UnseenSeverity):
    """1 - LCA-hF of the top-1 class on unseen-language sentences."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(1 - lca_f1(true_lineages, true_lineages[:num_classes]))


class KnownPathSeverity(KnownSeverity):
    """Path distance of the top-1 class in reference trees (e.g. asjp21) on known-language sentences."""

    def __init__(self, true_lineages, num_classes, trees):
        super().__init__(
            path_distance(true_lineages, true_lineages[:num_classes], trees)
        )


class UnseenPathSeverity(UnseenSeverity):
    """Path distance of the top-1 class in reference trees (e.g. asjp21) on unseen-language sentences."""

    def __init__(self, true_lineages, num_classes, trees):
        super().__init__(
            path_distance(true_lineages, true_lineages[:num_classes], trees)
        )


class UnseenSameFamily(UnseenSeverity):
    """Share of unseen-language sentences whose top-1 class is in their family (higher is better), per language."""

    def __init__(self, true_lineages, num_classes):
        super().__init__(same_family(true_lineages, true_lineages[:num_classes]))
