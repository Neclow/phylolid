"""fastText's supervised model in PyTorch, as ConLID's cross-entropy baseline, its phylogenetic losses and its
Lightning training module."""

import torch

from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import MetricCollection

from src.data.datasets import collate_features
from src.lid.metrics import KnownAccuracy, KnownMacroF1
from src.lid.tree import (
    IGNORE,
    hxe_layout,
    level_labels,
    tree_distances,
    tree_softmax_layout,
)
from src.lid.tree_losses import TreeLosses


class TreeSoftmax(nn.Module):
    """Output layer whose class scores are log-probabilities of choices down a tree (hierarchical softmax; fastText's
    hs on a binary tree): one score per unit of the layout (src/lid/tree.py, tree_softmax_layout), a softmax among
    each unit's siblings, and a class's log-probability the sum along its path from the root. They sum to 1 over the
    classes, so cross-entropy and softmax treat them as logits."""

    # The layout, saved with the checkpoint
    layout: torch.Tensor
    groups: torch.Tensor

    def __init__(self, dim, layout, groups):
        super().__init__()
        self.linear = nn.Linear(dim, layout.shape[1])
        self.register_buffer("layout", layout)
        self.register_buffer("groups", groups)

    def forward(self, embeddings):
        scores = self.linear(embeddings)
        groups = self.groups.expand_as(scores)
        # Log-softmax among siblings, from each group's maximum for stability; there are fewer groups than units
        top = torch.full_like(scores, float("-inf")).scatter_reduce(
            1, groups, scores.detach(), "amax"
        )
        scores = scores - top.gather(1, groups)
        total = torch.zeros_like(scores).scatter_add(1, groups, scores.exp())
        return (scores - total.log().gather(1, groups)) @ self.layout.T


class FastText(nn.Module):
    """Mean of a text's feature embeddings, then a linear layer over the classes, or another output layer. The
    embeddings' gradients are sparse: only the rows of a batch's features, as fastText updates them.
    """

    def __init__(self, n_features, n_classes, dim, output=None):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_features, dim, mode="mean", sparse=True)
        self.linear = nn.Linear(dim, n_classes) if output is None else output

    def forward(self, ids, offsets):
        return self.linear(self.embedding(ids, offsets))


def soft_targets(lineages, lam, beta, branch_lengths="levels"):
    """Training target of each class (rows) over the classes: the class keeps 1 - lam, and lam goes to the other classes
    of its family, by exp(-beta * their tree distance) (src/lid/tree.py, tree_distances, with these branch lengths).
    One-hot (cross-entropy) when lam is 0, and for a class without relatives: an isolate, or a member of a
    pseudo-family."""
    distances = tree_distances(lineages, branch_lengths)
    related = torch.isfinite(distances) & ~torch.eye(len(lineages), dtype=torch.bool)
    targets = torch.eye(len(lineages))
    rows = related.any(dim=1)
    if lam > 0 and rows.any():
        shares = torch.softmax(
            torch.where(related, -beta * distances, float("-inf"))[rows], dim=1
        )
        targets[rows] = (1 - lam) * targets[rows] + lam * shares
    return targets


# Glottolog levels the phylo-nn heads predict: the family (H0) and the ancestor three levels below it (H3)
PHYLONN_LEVELS = [0, 3]
# Losses that can be added to cross-entropy, each weighted by <name>_weight
EXTRA_LOSSES = ["patristic", "quartet", "phylonn"]
# GradNorm (Chen et al. 2018): its asymmetry and the learning rate of the losses' weights (the paper's defaults)
GRADNORM_ALPHA = 1.5
GRADNORM_LR = 0.025


def masked_cross_entropy(logits, labels):
    """Cross-entropy over the labels that are not IGNORE (0 when there is none)."""
    known = labels != IGNORE
    if not known.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[known], labels[known])


class LitFastText(LightningModule):
    """Trains FastText with Adam (SparseAdam for the embedding table) and a linearly decaying learning rate (ConLID's
    LIDCE, there with dense AdamW), against soft_targets (lam 0: cross-entropy), plus patristic_weight and
    quartet_weight times the TreeLosses of the sentence embeddings, plus phylonn_weight times the phylo-nn heads, which
    predict each sentence's Glottolog groups at PHYLONN_LEVELS from its embedding. With gradnorm, the weights of
    cross-entropy and these extra losses are learned (gradnorm_loss), all from 1 and summing to the number of losses.
    With tree_softmax, the output layer is a TreeSoftmax over the lineages' tree. With hxe_alpha > 0, the loss is the
    hierarchical cross-entropy (hxe_loss) instead, against the true classes. The soft targets and the tree losses
    measure the tree with branch_lengths (src/lid/tree.py, BRANCH_LENGTHS).
    The classes' lineages are needed for training only. spelling is only saved with the checkpoint: how its features
    read the text (src/lid/features.py, SPELLINGS), for the predictions to read it alike.
    """

    # Buffer of the training targets, typed as torchmetrics does, and the tree losses (training only)
    targets: torch.Tensor
    phylonn_labels: torch.Tensor
    hxe_members: torch.Tensor
    hxe_paths: torch.Tensor
    tree_losses: TreeLosses
    first_losses: torch.Tensor | None

    def __init__(
        self,
        n_features,
        classes,
        dim,
        lr,
        lam=0.0,
        beta=0.5,
        patristic_weight=0.0,
        quartet_weight=0.0,
        phylonn_weight=0.0,
        phylonn_sizes=None,
        tree_softmax=False,
        tree_softmax_units=0,
        spelling="raw",
        gradnorm=False,
        branch_lengths="levels",
        hxe_alpha=0.0,
        lineages=None,
    ):
        super().__init__()
        # Two optimizers (configure_optimizers), stepped in training_step
        self.automatic_optimization = False
        if lineages is not None:
            labels = [level_labels(lineages, level) for level in PHYLONN_LEVELS]
            # Numbers of groups per level: the heads' sizes, saved so that a checkpoint loads without lineages
            phylonn_sizes = [max(max(level), 0) + 1 for level in labels]
        output = None
        if tree_softmax:
            # The layout from the lineages, or zeros of its saved size, which the checkpoint fills
            if lineages is not None:
                layout, groups = tree_softmax_layout(lineages)
                tree_softmax_units = len(groups)
            else:
                layout = torch.zeros(len(classes), tree_softmax_units)
                groups = torch.zeros(tree_softmax_units, dtype=torch.long)
            output = TreeSoftmax(dim, layout, groups)
        # Saved with the checkpoint, classes (glottocodes in output order) included
        self.save_hyperparameters(ignore=["lineages"])
        if phylonn_sizes is not None:
            self.phylonn_heads = nn.ModuleList(
                nn.Linear(dim, size) for size in phylonn_sizes
            )
        if lineages is not None:
            self.register_buffer(
                "phylonn_labels", torch.tensor(labels), persistent=False
            )
            self.register_buffer(
                "targets",
                soft_targets(lineages, lam, beta, branch_lengths),
                persistent=False,
            )
            self.tree_losses = TreeLosses(tree_distances(lineages, branch_lengths), dim)
            members, paths = hxe_layout(lineages)
            self.register_buffer("hxe_members", members, persistent=False)
            self.register_buffer("hxe_paths", paths, persistent=False)
        self.model = FastText(n_features, len(classes), dim, output)
        self.loss_fn = nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            [KnownAccuracy(len(classes)), KnownMacroF1(len(classes))],
            prefix="valid_",
        )
        # The extra losses in use (weight > 0); with gradnorm, their weights and cross-entropy's are learned, from 1
        self.extra_losses = [
            name for name in EXTRA_LOSSES if self.hparams[f"{name}_weight"]
        ]
        if gradnorm:
            self.loss_weights = nn.Parameter(torch.ones(1 + len(self.extra_losses)))
            # Cross-entropy's and the extra losses' values at the first training step (gradnorm_loss)
            self.first_losses = None

    def forward(self, ids, offsets):
        return self.model(ids, offsets)

    def gradnorm_loss(self, losses):
        """GradNorm (Chen et al. 2018) over cross-entropy, losses[0], and the extra losses: each loss's weighted gradient
        on the embedding table (the layer they share) is pulled to the mean of all of them, times its relative training
        rate (its loss over its first, over the mean of these ratios) to the power GRADNORM_ALPHA. A loss of the weights
        only: the gradient sizes and their target are constants."""
        values = torch.stack(losses).detach()
        if self.first_losses is None:
            self.first_losses = values
        # Sparse gradients: their rows summed (coalesce) before the size
        table = self.model.embedding.weight
        sizes = torch.stack(
            [
                torch.autograd.grad(loss, table, retain_graph=True)[0]
                .coalesce()
                .values()
                .norm()
                for loss in losses
            ]
        )
        # A weighted loss's gradient is its weight times the loss's: no second-order gradient needed
        gradients = self.loss_weights * sizes
        # A loss that started at 0 (nothing to score, e.g. no quartet) gives no rate: 1
        rates = torch.where(
            self.first_losses > 0, values / self.first_losses, torch.ones_like(values)
        )
        target = gradients.mean().detach() * (rates / rates.mean()) ** GRADNORM_ALPHA
        return (gradients - target).abs().sum()

    def hxe_loss(self, logits, target):
        """Hierarchical cross-entropy (Bertinetto et al. 2020): minus the sum, down the true class's path, of
        exp(-hxe_alpha * depth) log p(node | its parent), a node's probability the sum of its classes' (the root: 1, the
        families: depth 1). The last term is the class's own, log p(class | its node), 0 unless classes lie below it (a
        language above its dialect). Cross-entropy when hxe_alpha is 0, as the logs add up to log p(class).
        """
        log_probs = logits.log_softmax(dim=-1)
        paths = self.hxe_paths[target]
        own = log_probs.gather(1, target.unsqueeze(1))
        # Each path node's log-probability; past a path's end, the class's own (no term)
        held = self.hxe_members[paths.clamp(min=0)]
        nodes = log_probs.unsqueeze(1).masked_fill(~held, float("-inf")).logsumexp(-1)
        nodes = torch.where(paths >= 0, nodes, own)
        chain = torch.cat([torch.zeros_like(own), nodes, own], dim=1)
        depths = torch.arange(1, chain.shape[1], device=logits.device)
        weights = torch.exp(-self.hparams["hxe_alpha"] * depths)
        return -((chain[:, 1:] - chain[:, :-1]) * weights).sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        embeddings = self.model.embedding(batch["ids"], batch["offsets"])
        logits = self.model.linear(embeddings)
        # Against each class's soft target, or hierarchical cross-entropy; dev scores stay against the true class,
        # comparable across lam and hxe_alpha
        if self.hparams["hxe_alpha"]:
            ce = self.hxe_loss(logits, batch["target"])
        else:
            ce = self.loss_fn(logits, self.targets[batch["target"]])
        # Always computed, as a measure of how tree-like the embeddings are; with a gradient only when used
        with torch.set_grad_enabled(
            "patristic" in self.extra_losses or "quartet" in self.extra_losses
        ):
            patristic, quartet = self.tree_losses(embeddings, batch["target"])
        # The phylo-nn heads, one per level; when not used, they are trained on the embeddings detached (probes,
        # outside the model's loss), so they only measure how much of the Glottolog groups the embeddings hold
        inputs = embeddings if "phylonn" in self.extra_losses else embeddings.detach()
        phylonn = [
            masked_cross_entropy(head(inputs), labels[batch["target"]])
            for head, labels in zip(self.phylonn_heads, self.phylonn_labels)
        ]
        values = {"patristic": patristic, "quartet": quartet, "phylonn": sum(phylonn)}
        names = ["ce", *self.extra_losses]
        losses = [ce, *(values[name] for name in self.extra_losses)]
        # The losses' weights: cross-entropy's 1 and the extra losses' as given, or all learned (GradNorm), as constants
        # in the model's loss
        if self.hparams["gradnorm"]:
            weights = self.loss_weights.detach()
            gradnorm = self.gradnorm_loss(losses)
        else:
            weights = [1.0] + [
                self.hparams[f"{name}_weight"] for name in self.extra_losses
            ]
            gradnorm = 0.0
        loss = sum(weight * value for weight, value in zip(weights, losses))
        probes = 0.0 if "phylonn" in self.extra_losses else values["phylonn"]
        # The number of texts (Lightning would infer it from the ids)
        n = len(batch["target"])
        self.log_dict(
            {
                "train_loss": loss,
                "train_ce_loss": ce,
                "train_patristic_loss": patristic,
                "train_quartet_loss": quartet,
            }
            | {
                f"train_phylonn_h{level}_loss": value
                for level, value in zip(PHYLONN_LEVELS, phylonn)
            }
            | {f"train_{name}_weight": weight for name, weight in zip(names, weights)},
            batch_size=n,
        )
        # train_loss is the model's objective; the probes and the weights (GradNorm) train on their own losses
        optimizers, schedulers = self.optimizers(), self.lr_schedulers()
        # Lists: configure_optimizers returns two optimizers (three with GradNorm) and two schedulers
        assert isinstance(optimizers, list) and isinstance(schedulers, list)
        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss + probes + gradnorm)
        # Only the dense optimizer steps through Lightning, so that it counts one step per batch (global_step)
        sparse, dense, *weights_optimizer = optimizers
        sparse.optimizer.step()
        dense.step()
        for optimizer in weights_optimizer:
            optimizer.optimizer.step()
            # GradNorm's weights kept positive and summing to the number of losses
            with torch.no_grad():
                self.loss_weights.clamp_(min=1e-8)
                self.loss_weights.mul_(len(self.loss_weights) / self.loss_weights.sum())
        for scheduler in schedulers:
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        logits = self(batch["ids"], batch["offsets"])
        loss = self.loss_fn(logits, batch["target"])
        self.log("valid_loss", loss, batch_size=len(batch["target"]))
        # Metric objects, so that Lightning computes them over the whole dev split
        self.valid_metrics.update(logits, batch["target"])
        self.log_dict(self.valid_metrics, batch_size=len(batch["target"]))

    def configure_optimizers(self):
        # ConLID's settings (appendix B.2): AdamW without weight decay, the learning rate falls linearly to 0 over
        # training. The embedding table, whose gradients are sparse, takes SparseAdam, which updates only the rows a
        # batch touches; without weight decay, AdamW is Adam, and SparseAdam its sparse form
        lr, betas, eps = self.hparams["lr"], (0.9, 0.999), 1e-6
        embedding = self.model.embedding.weight
        # GradNorm's weights take their own optimizer
        own = [embedding] + ([self.loss_weights] if self.hparams["gradnorm"] else [])
        optimizers = [
            torch.optim.SparseAdam([embedding], lr=lr, betas=betas, eps=eps),
            torch.optim.AdamW(
                [p for p in self.parameters() if all(p is not q for q in own)],
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=0,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=int(self.trainer.estimated_stepping_batches),
            )
            for optimizer in optimizers
        ]
        # GradNorm's weights at a constant learning rate
        if self.hparams["gradnorm"]:
            optimizers.append(torch.optim.Adam([self.loss_weights], lr=GRADNORM_LR))
        return optimizers, schedulers


class Predictor:
    """A FastText we trained, from its Lightning checkpoint, behind the zoo models' interface (src/models/pretrained.py):
    labels() (the class glottocodes) and predict_proba()."""

    def __init__(self, checkpoint, features, device="cpu"):
        self.model_id = checkpoint
        self.lit = LitFastText.load_from_checkpoint(checkpoint, map_location=device)
        # The vocabulary must be the one the model was trained with
        assert features.size == self.lit.hparams["n_features"], "another vocabulary"
        self.model = self.lit.model.eval()
        self.features = features
        self.device = device

    def labels(self):
        return list(self.lit.hparams["classes"])

    def predict_proba(self, texts):
        batch = collate_features([(self.features(text), 0) for text in texts])
        with torch.no_grad():
            logits = self.model(
                batch["ids"].to(self.device), batch["offsets"].to(self.device)
            )
        return logits.softmax(dim=-1).cpu().numpy()
