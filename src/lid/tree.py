"""Trees over the language classes (Glottolog's, or fastText's Huffman tree) and what the training methods take from them."""

import heapq
import random

import torch

from ete4 import Tree

from src._config import PSEUDO_FAMILIES

# Label of a class a hierarchy head ignores (CrossEntropyLoss's default ignore_index)
IGNORE = -100
# Tree softmax: the child through which a class with another class below it is chosen (a language above its dialect)
STOP = "<stop>"
# Branch lengths of the tree distances: one per Glottolog level, or exponential and ultrametric (tree_distances)
BRANCH_LENGTHS = ["levels", "exponential"]


def shared(a, b):
    """Nodes two lineages (family ... languoid) share from the family down: their lowest common ancestor's depth + 1."""
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    return common


def levels(a, b):
    """Glottolog levels between two lineages (family ... languoid): up from one to their lowest common ancestor, then
    down to the other (sisters: 2)."""
    return len(a) + len(b) - 2 * shared(a, b)


def tree_distances(lineages, branch_lengths="levels"):
    """Path distances between the classes of one family, NaN between classes of different families or of a
    pseudo-family (no tree relates them); 0 on the diagonal. Branch lengths: one per Glottolog level, or exponential
    (the family's first branches 1/2, the next 1/4, and so on, as glottoTrees after Venditti et al. 2010) with every
    class stretched to the depth of the family's deepest, D (ultrametric), so that two classes whose lowest common
    ancestor is at depth k (the family: 0) are 2 (2^-k - 2^-D) apart."""
    deepest = {}
    for lineage in lineages:
        deepest[lineage[0]] = max(deepest.get(lineage[0], 0), len(lineage) - 1)
    distances = torch.full((len(lineages), len(lineages)), float("nan"))
    for a, lineage in enumerate(lineages):
        distances[a, a] = 0
        if lineage[0] in PSEUDO_FAMILIES:
            continue
        for b, other in enumerate(lineages):
            if b != a and other[0] == lineage[0]:
                if branch_lengths == "levels":
                    distances[a, b] = levels(lineage, other)
                else:
                    k = shared(lineage, other) - 1
                    distances[a, b] = 2 * (2.0**-k - 2.0 ** -deepest[lineage[0]])
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


def huffman_lineages(classes, counts, seed):
    """Lineages of the binary Huffman tree fastText's hierarchical softmax builds from the label counts (src/loss.cc):
    the two least frequent nodes merge until one is left, so frequent classes sit near the root. Ties are broken at
    random (seed), as fastText's order of tied labels is arbitrary; the classes' alphabetical order would group
    related glottocodes. Internal nodes are named huffman<i>, the root huffman0."""
    rng = random.Random(seed)
    heap = [(count, rng.random(), name) for name, count in zip(classes, counts)]
    heapq.heapify(heap)
    parent = {}
    while len(heap) > 1:
        (count_a, _, a), (count_b, _, b) = heapq.heappop(heap), heapq.heappop(heap)
        # Nodes left after the merge, so each name is used once
        node = f"huffman{len(heap)}"
        parent[a] = parent[b] = node
        heapq.heappush(heap, (count_a + count_b, rng.random(), node))
    lineages = []
    for name in classes:
        path = [name]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        lineages.append(tuple(reversed(path)))
    return lineages


def hxe_layout(lineages):
    """For hierarchical cross-entropy (src/lid/model.py, hxe_loss): the tree's nodes below the root as the classes each
    holds, a (nodes, classes) mask, and each class's path of nodes from the root down to its own, a (classes, longest
    path) matrix of node indices, -1 past its end. As in tree_softmax_layout, the families hang from one root, and so do
    the pseudo-families' members."""
    paths = [
        (lineage[-1],) if lineage[0] in PSEUDO_FAMILIES else lineage
        for lineage in lineages
    ]
    # Nodes by their path from the root, as in tree_softmax_layout
    nodes = {}
    for path in paths:
        for depth in range(1, len(path) + 1):
            nodes.setdefault(path[:depth], len(nodes))
    members = torch.zeros(len(nodes), len(paths), dtype=torch.bool)
    order = torch.full((len(paths), max(len(path) for path in paths)), -1)
    for c, path in enumerate(paths):
        for depth in range(1, len(path) + 1):
            members[nodes[path[:depth]], c] = True
            order[c, depth - 1] = nodes[path[:depth]]
    return members, order


def tree_softmax_layout(lineages):
    """The tree softmax's layout (src/lid/model.py, TreeSoftmax) over the lineages' tree: its units, the nodes that
    have siblings (an only child is chosen with probability 1, so it needs no score), as a (classes, units) matrix
    with 1 where a unit is on a class's path from the root, and each unit's sibling group. The families hang from one
    root, and so do the pseudo-families' members, as they are not genealogical groups; a class with another class
    below it is reached through a STOP child, so that every class is a leaf."""
    paths = [
        (lineage[-1],) if lineage[0] in PSEUDO_FAMILIES else lineage
        for lineage in lineages
    ]
    inner = {path[:depth] for path in paths for depth in range(1, len(path))}
    paths = [path + (STOP,) if path in inner else path for path in paths]
    # Children of each node, in order of appearance; the root is ()
    children = {}
    for path in paths:
        for depth in range(1, len(path) + 1):
            children.setdefault(path[: depth - 1], {})[path[:depth]] = None
    units, groups = {}, []
    siblings = [nodes for nodes in children.values() if len(nodes) > 1]
    for group, nodes in enumerate(siblings):
        for node in nodes:
            units[node] = len(units)
            groups.append(group)
    layout = torch.zeros(len(paths), len(units))
    for i, path in enumerate(paths):
        for depth in range(1, len(path) + 1):
            if path[:depth] in units:
                layout[i, units[path[:depth]]] = 1
    return layout, torch.tensor(groups)
