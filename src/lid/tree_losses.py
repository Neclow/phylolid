"""Losses that pull the classes' embedding centroids towards the Glottolog tree, after the phylogeny-guided-lid project."""

import torch
import torch.nn.functional as F

from torch import nn


def pearson(a, b, eps=1e-8):
    """Pearson correlation of two vectors; eps under the square roots keeps the gradient finite at zero variance."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (
        torch.sqrt((a * a).sum() + eps) * torch.sqrt((b * b).sum() + eps)
    )


class TreeLosses(nn.Module):
    """Two losses on per-class centroids of the sentence embeddings, against tree distances within families (NaN
    elsewhere, src/lid/tree.py):

    - tree: 1 - Pearson correlation between the centroids' distances and the tree distances, over related pairs;
    - quartet (Phyla): for four related classes, cross-entropy of the three pairings, scored by the centroids'
      distances, against the pairing the tree implies (four-point condition).

    Centroids are running averages (EMA) of each class's embeddings; the classes of the batch enter blended with their
    batch mean, so that the gradient reaches the embeddings, the others as constants. No state is saved with a
    checkpoint.
    """

    # Buffers, typed as torchmetrics does
    distances: torch.Tensor
    related: torch.Tensor
    pairs: torch.Tensor
    centroids: torch.Tensor
    seen: torch.Tensor

    def __init__(self, distances, dim, momentum=0.9, n_quartets=512, temperature=1.0):
        super().__init__()
        self.momentum = momentum
        self.n_quartets = n_quartets
        self.temperature = temperature
        n = len(distances)
        related = torch.isfinite(distances) & ~torch.eye(n, dtype=torch.bool)
        # Related pairs, once: (i, j) with i < j
        pairs = torch.triu(related, diagonal=1).nonzero().T
        self.register_buffer("distances", distances, persistent=False)
        self.register_buffer("related", related, persistent=False)
        self.register_buffer("pairs", pairs, persistent=False)
        self.register_buffer("centroids", torch.zeros(n, dim), persistent=False)
        self.register_buffer("seen", torch.zeros(n, dtype=torch.bool), persistent=False)

    def forward(self, embeddings, targets):
        """Tree and quartet losses of a batch of sentence embeddings (B, dim) with their classes (B,)."""
        n = len(self.distances)
        sums = embeddings.new_zeros(n, embeddings.size(1)).index_add(
            0, targets, embeddings
        )
        counts = torch.bincount(targets, minlength=n)
        present = counts > 0
        means = sums / counts.clamp(min=1).unsqueeze(1)
        # Copies: the buffers are updated below, while the gradient still needs their values
        seen, previous = self.seen.clone(), self.centroids.clone()
        # A class seen before blends its running centroid with this batch's mean; a new one starts at its mean
        blended = torch.where(
            seen.unsqueeze(1),
            self.momentum * previous + (1 - self.momentum) * means,
            means,
        )
        centroids = torch.where(present.unsqueeze(1), blended, previous)
        with torch.no_grad():
            self.centroids.copy_(centroids)
            self.seen |= present
        # Distances between all centroids at once, from their dot products; eps under the square root: a finite
        # gradient when two centroids coincide (at initialisation)
        squares = (centroids * centroids).sum(dim=1)
        products = centroids @ centroids.T
        squared = squares.unsqueeze(1) + squares.unsqueeze(0) - 2 * products
        embedded = torch.sqrt(squared.clamp(min=0) + 1e-8)
        return self.tree_loss(embedded), self.quartet_loss(embedded, present)

    def tree_loss(self, embedded):
        i, j = self.pairs
        known = self.seen[i] & self.seen[j]
        # Correlation is undefined over fewer than three pairs
        if known.sum() < 3:
            return embedded.sum() * 0.0
        i, j = i[known], j[known]
        return 1 - pearson(embedded[i, j], self.distances[i, j])

    def quartet_loss(self, embedded, present):
        # Each quartet: a class of the batch (which carries the gradient) and three of its relatives seen so far
        candidates = self.related & self.seen.unsqueeze(0)
        anchors = (present & (candidates.sum(dim=1) >= 3)).nonzero().squeeze(1)
        if len(anchors) == 0:
            return embedded.sum() * 0.0
        device = embedded.device
        anchors = anchors[
            torch.randint(len(anchors), (self.n_quartets,), device=device)
        ]
        keys = torch.rand(self.n_quartets, len(self.distances), device=device)
        others = keys.masked_fill(~candidates[anchors], -1).topk(3, dim=1).indices
        q = torch.cat([anchors.unsqueeze(1), others], dim=1)

        # The three pairings ab|cd, ac|bd, ad|bc: sums of the distances within their two pairs
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        scores = torch.stack(
            [
                embedded[q[:, a], q[:, b]] + embedded[q[:, c], q[:, d]]
                for (a, b), (c, d) in pairings
            ],
            dim=1,
        )
        sums = torch.stack(
            [
                self.distances[q[:, a], q[:, b]] + self.distances[q[:, c], q[:, d]]
                for (a, b), (c, d) in pairings
            ],
            dim=1,
        )
        # The tree's pairing has the smallest sum; none when the smallest is tied (four classes below one node)
        ordered = sums.sort(dim=1).values
        unique = ordered[:, 0] < ordered[:, 1]
        if not unique.any():
            return embedded.sum() * 0.0
        logits = -scores[unique]
        logits = (logits - logits.mean(dim=1, keepdim=True)) / self.temperature
        return F.cross_entropy(logits, sums[unique].argmin(dim=1))
