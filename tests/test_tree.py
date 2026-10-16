"""Unit tests for the Glottolog tree distances in src/lid/tree.py."""

from itertools import permutations

import torch

from ete4 import Tree

from src.lid.tree import (
    IGNORE,
    level_labels,
    levels,
    newick,
    shuffle_classes,
    tree_distances,
)

# Sisters a and b, a language l above its dialect d, an isolate e, two members of a pseudo-family
LINEAGES = [
    ("f", "g", "a"),
    ("f", "g", "b"),
    ("f", "l"),
    ("f", "l", "d"),
    ("e",),
    ("arti1236", "p"),
    ("arti1236", "q"),
]


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


def test_level_labels():
    lineages = [
        # Two languages below one H3 ancestor (i), one below another (j)
        ("f", "g", "h", "i", "a"),
        ("f", "g", "h", "i", "b"),
        ("f", "g", "h", "j", "c"),
        # A shallower lineage: no H3 ancestor, so the language itself
        ("f", "d"),
        # An isolate: its own family; a pseudo-family member: ignored
        ("z",),
        ("arti1236", "p"),
    ]
    assert level_labels(lineages, 0) == [0, 0, 0, 0, 1, IGNORE]
    assert level_labels(lineages, 3) == [0, 0, 1, 2, 3, IGNORE]


def test_shuffle_classes_keeps_the_tree():
    shuffled = shuffle_classes(LINEAGES, seed=1)
    assert shuffled != LINEAGES
    # Each class keeps its name; the pseudo-family's members keep their lineages
    assert [lineage[-1] for lineage in shuffled] == [
        lineage[-1] for lineage in LINEAGES
    ]
    assert shuffled[5:] == LINEAGES[5:]
    # The same tree with the five real classes permuted: the distances between their new places are the true ones
    true = tree_distances(LINEAGES)[:5, :5]
    new = tree_distances(shuffled)[:5, :5]
    assert any(
        torch.equal(new.nan_to_num(-1), true[p][:, p].nan_to_num(-1))
        for p in map(list, permutations(range(5)))
    )


def test_newick():
    lines = newick(LINEAGES).split("\n")
    # One tree per real family, the language l above its dialect d; the pseudo-family left out
    assert lines == ["((a,b)g,(d)l)f;", "e;"]
    names = [node.name for line in lines for node in Tree(line, parser=1).traverse()]
    assert sorted(names) == ["a", "b", "d", "e", "f", "g", "l"]
