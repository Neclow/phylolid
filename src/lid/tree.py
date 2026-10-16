"""Distances and groups of language classes in the Glottolog tree, for the phylogenetic training losses."""

import random

import torch

from ete4 import Tree

from src._config import PSEUDO_FAMILIES

# Label of a class a hierarchy head ignores (CrossEntropyLoss's default ignore_index)
IGNORE = -100


def levels(a, b):
    """Glottolog levels between two lineages (family ... languoid): up from one to their lowest common ancestor, then
    down to the other (sisters: 2)."""
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    return len(a) + len(b) - 2 * common


def tree_distances(lineages):
    """Glottolog levels between the classes of one family, NaN between classes of different families or of a
    pseudo-family (no tree relates them); 0 on the diagonal."""
    distances = torch.full((len(lineages), len(lineages)), float("nan"))
    for a, lineage in enumerate(lineages):
        distances[a, a] = 0
        if lineage[0] in PSEUDO_FAMILIES:
            continue
        for b, other in enumerate(lineages):
            if b != a and other[0] == lineage[0]:
                distances[a, b] = levels(lineage, other)
    return distances


def level_labels(lineages, level):
    """Each class's group at a Glottolog level, for the phylo-nn heads: its ancestor at that level (0: its family), or
    the class itself when its lineage is shallower, so that every class has one, as every species has an ancestor at
    each level in Phylo-NN (Elhamod et al. 2023); IGNORE for the pseudo-families, which are not genealogical. Groups are
    numbered in order of appearance."""
    ids = {}
    labels = []
    for lineage in lineages:
        if lineage[0] in PSEUDO_FAMILIES:
            labels.append(IGNORE)
        else:
            labels.append(
                ids.setdefault(lineage[min(level, len(lineage) - 1)], len(ids))
            )
    return labels


def shuffle_classes(lineages, seed):
    """The lineages with the classes of the real families shuffled over their places in the tree, for a control with
    the tree's shape but random relatives. A class is renamed wherever it occurs, as a class can be the ancestor of
    another (a language of its dialect); the pseudo-families' members keep their lineages.
    """
    classes = [lineage[-1] for lineage in lineages if lineage[0] not in PSEUDO_FAMILIES]
    shuffled = classes.copy()
    random.Random(seed).shuffle(shuffled)
    rename = dict(zip(classes, shuffled))
    # Each node keeps its place under its new name, and the class of that name takes its lineage
    renamed = [
        tuple(rename.get(node, node) for node in lineage) for lineage in lineages
    ]
    by_class = {lineage[-1]: lineage for lineage in renamed}
    return [by_class[lineage[-1]] for lineage in lineages]


def newick(lineages):
    """The lineages' tree in Newick, one line per family, every node named by its glottocode (ete4's parser 1, as the
    reference trees); the pseudo-families, not trees, are left out."""
    # Nodes by their path from the family; a lineage's ancestors come before it
    nodes = {}
    for lineage in lineages:
        if lineage[0] in PSEUDO_FAMILIES:
            continue
        for depth in range(1, len(lineage) + 1):
            path = lineage[:depth]
            if path not in nodes:
                nodes[path] = (
                    Tree({"name": path[-1]})
                    if depth == 1
                    else nodes[path[:-1]].add_child(name=path[-1])
                )
    return "\n".join(
        node.write(parser=1, format_root_node=True)
        for path, node in nodes.items()
        if len(path) == 1
    )
