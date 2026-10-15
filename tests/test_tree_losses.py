"""Unit tests for the tree losses in src/lid/tree_losses.py."""

import numpy as np
import torch

from src.lid.tree import tree_distances
from src.lid.tree_losses import TreeLosses, pearson

# One family: a and b sisters, c and d sisters (pairing ab|cd); e alone in another family
LINEAGES = [
    ("f", "g", "a"),
    ("f", "g", "b"),
    ("f", "h", "c"),
    ("f", "h", "d"),
    ("x", "e"),
]


def test_pearson_matches_numpy():
    a, b = torch.randn(50), torch.randn(50)
    expected = torch.tensor(
        np.corrcoef(a.numpy(), b.numpy())[0, 1], dtype=torch.float32
    )
    assert torch.isclose(pearson(a, b), expected, atol=1e-5)


def test_centroids_are_running_means():
    losses = TreeLosses(tree_distances(LINEAGES), dim=2, momentum=0.9)
    losses(torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 0, 2]))
    # First sight: the batch mean
    assert torch.allclose(losses.centroids[0], torch.tensor([2.0, 0.0]))
    assert losses.seen.tolist() == [True, False, True, False, False]
    # Then blended with the running centroid
    losses(torch.tensor([[12.0, 0.0]]), torch.tensor([0]))
    assert torch.allclose(losses.centroids[0], torch.tensor([0.9 * 2 + 0.1 * 12, 0.0]))


def test_tree_like_centroids_score_better():
    distances = tree_distances(LINEAGES)
    targets = torch.arange(5)
    # Sisters close together, or the pairs swapped (a with c, b with d)
    tree_like = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0], [0.0, 9.0]]
    )
    swapped = tree_like[[0, 2, 1, 3, 4]]
    good = TreeLosses(distances, dim=2, temperature=0.1)(tree_like, targets)
    bad = TreeLosses(distances, dim=2, temperature=0.1)(swapped, targets)
    assert good[0] < 0.05 < bad[0]
    assert good[1] < bad[1]
