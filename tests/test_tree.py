"""Unit tests for the Glottolog tree distances in src/lid/tree.py."""

import torch

from src.lid.tree import levels, tree_distances


def test_levels():
    # Sisters, a dialect below its language, and a language two branches away
    assert levels(("f", "g", "a"), ("f", "g", "b")) == 2
    assert levels(("f", "l"), ("f", "l", "d")) == 1
    assert levels(("f", "g", "a"), ("f", "c")) == 3


def test_tree_distances():
    lineages = [
        ("f", "g", "a"),
        ("f", "g", "b"),
        ("f", "c"),
        ("h", "d"),
        ("arti1236", "p"),
        ("arti1236", "q"),
    ]
    distances = tree_distances(lineages)
    assert distances[0, 1] == distances[1, 0] == 2
    assert distances[0, 2] == 3
    assert torch.equal(distances.diagonal(), torch.zeros(6))
    # Other families and the members of a pseudo-family: no distance
    assert distances[0, 3].isnan() and distances[3, 0].isnan()
    assert distances[4, 5].isnan()
