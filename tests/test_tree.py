"""Unit tests for the trees over the language classes in src/lid/tree.py."""

from itertools import permutations

import torch

from ete4 import Tree

from src.lid.tree import (
    IGNORE,
    huffman_lineages,
    hxe_layout,
    level_labels,
    levels,
    newick,
    shuffle_classes,
    tree_distances,
    tree_softmax_layout,
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


def test_exponential_tree_distances():
    # Family f's deepest classes at depth 2 (D); a dialect d below its language c
    lineages = [
        ("f", "g", "a"),
        ("f", "g", "b"),
        ("f", "c"),
        ("f", "c", "d"),
        ("h", "e"),
        ("arti1236", "p"),
        ("arti1236", "q"),
    ]
    distances = tree_distances(lineages, "exponential")
    # Sisters below g (depth 1): 2 (1/2 - 1/4); across f's first split (depth 0): 2 (1 - 1/4)
    assert distances[0, 1] == distances[1, 0] == 0.5
    assert distances[0, 2] == distances[0, 3] == 1.5
    # Ultrametric: the dialect is as far from its language as sisters are, c being stretched to depth 2
    assert distances[2, 3] == 0.5
    assert torch.equal(distances.diagonal(), torch.zeros(7))
    assert distances[0, 4].isnan() and distances[5, 6].isnan()


def test_hxe_layout():
    members, paths = hxe_layout(LINEAGES)
    # Nodes in order of appearance: f, g, a, b, l, d, e, p, q (the pseudo-family's members hang from the root)
    assert paths.tolist() == [
        [0, 1, 2],
        [0, 1, 3],
        [0, 4, -1],
        [0, 4, 5],
        [6, -1, -1],
        [7, -1, -1],
        [8, -1, -1],
    ]
    # Each node holds the classes at or below it: f all five of its family, the language l itself and its dialect d
    assert members[0].tolist() == [True] * 4 + [False] * 3
    assert members[4].tolist() == [False, False, True, True, False, False, False]
    assert members.sum(dim=1).tolist() == [4, 2, 1, 1, 2, 1, 1, 1, 1]


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


def test_huffman_lineages():
    classes = ["a", "b", "c", "d", "e"]
    lineages = huffman_lineages(classes, [8, 4, 2, 1, 1], seed=0)
    # One root; each class ends its own lineage, the frequent ones nearer the root
    assert {lineage[0] for lineage in lineages} == {"huffman0"}
    assert [lineage[-1] for lineage in lineages] == classes
    assert [len(lineage) - 1 for lineage in lineages] == [1, 2, 3, 4, 4]
    # Binary: every internal node has two children
    children = {}
    for lineage in lineages:
        for depth in range(1, len(lineage)):
            children.setdefault(lineage[depth - 1], set()).add(lineage[depth])
    assert all(len(nodes) == 2 for nodes in children.values())


def test_huffman_ties_follow_the_seed():
    # Equal counts: a balanced tree, its pairs set by the seed rather than the classes' order
    classes = list("abcdefgh")
    trees = [huffman_lineages(classes, [1] * 8, seed) for seed in range(2)]
    assert all(len(lineage) == 4 for tree in trees for lineage in tree)
    # Sister pairs: the classes below each class's parent
    pairs = [
        {
            frozenset(c for c, other in zip(classes, tree) if other[-2] == lineage[-2])
            for lineage in tree
        }
        for tree in trees
    ]
    assert pairs[0] != pairs[1]


def test_tree_softmax_layout():
    layout, groups = tree_softmax_layout(LINEAGES)
    # Units by sibling group: the root's children (f, the isolate e, the pseudo-family's p and q), f's (g, l), g's
    # (a, b), l's (its STOP child, d)
    assert groups.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    paths = [row.nonzero().flatten().tolist() for row in layout]
    assert paths == [[0, 4, 6], [0, 4, 7], [0, 5, 8], [0, 5, 9], [1], [2], [3]]
