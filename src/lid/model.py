"""fastText's supervised model in PyTorch, as ConLID's cross-entropy baseline, its phylogenetic losses and its
Lightning training module."""

import torch

from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import MetricCollection

from src.data.datasets import collate_features
from src.lid.metrics import KnownAccuracy, KnownMacroF1
from src.lid.tree import IGNORE, level_labels, tree_distances, tree_softmax_layout
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


def soft_targets(lineages, lam, beta):
    """Training target of each class (rows) over the classes: the class keeps 1 - lam, and lam goes to the other classes
    of its family, by exp(-beta * levels between them). One-hot (cross-entropy) when lam is 0, and for a class without
    relatives: an isolate, or a member of a pseudo-family."""
    distances = tree_distances(lineages)
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
# GradNorm (Chen et al. 2018): its asymmetry (the paper's default) and the learning rate of the extra losses' log weights
GRADNORM_ALPHA = 1.5
GRADNORM_LR = 0.01


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
    predict each sentence's Glottolog groups at PHYLONN_LEVELS from its embedding. With gradnorm, the weights of these
    extra losses are learned (gradnorm_loss), from their given values. With tree_softmax, the output layer is a
    TreeSoftmax over the lineages' tree.
    The classes' lineages are needed for training only. spelling is only saved with the checkpoint: how its features
    read the text (src/lid/features.py, SPELLINGS), for the predictions to read it alike.
    """

    # Buffer of the training targets, typed as torchmetrics does, and the tree losses (training only)
    targets: torch.Tensor
    phylonn_labels: torch.Tensor
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
                "targets", soft_targets(lineages, lam, beta), persistent=False
            )
            self.tree_losses = TreeLosses(tree_distances(lineages), dim)
        self.model = FastText(n_features, len(classes), dim, output)
        self.loss_fn = nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            [KnownAccuracy(len(classes)), KnownMacroF1(len(classes))],
            prefix="valid_",
        )
        # The extra losses in use (weight > 0); with gradnorm, their weights are learned as logs, from the given ones
        self.extra_losses = [
            name for name in EXTRA_LOSSES if self.hparams[f"{name}_weight"]
        ]
        if gradnorm:
            weights = torch.tensor(
                [self.hparams[f"{name}_weight"] for name in self.extra_losses]
            )
            self.log_weights = nn.Parameter(weights.log())
            # Cross-entropy's and the extra losses' values at the first training step (gradnorm_loss)
            self.first_losses = None

    def forward(self, ids, offsets):
        return self.model(ids, offsets)

    def gradnorm_loss(self, embeddings, losses):
        """GradNorm (Chen et al. 2018) with cross-entropy, losses[0], as the fixed reference (weight 1): each extra loss's
        weighted gradient on the sentence embeddings is pulled to the mean of all of them, times its relative training
        rate (its loss over its first, over the mean of these ratios) to the power GRADNORM_ALPHA. A loss of the log
        weights only: the gradient sizes and their target are constants."""
        values = torch.stack(losses).detach()
        if self.first_losses is None:
            self.first_losses = values
        sizes = torch.stack(
            [
                torch.autograd.grad(loss, embeddings, retain_graph=True)[0].norm()
                for loss in losses
            ]
        )
        weights = torch.cat([sizes.new_ones(1), self.log_weights.exp()])
        gradients = weights * sizes
        # A loss that started at 0 (nothing to score, e.g. no quartet) gives no rate: 1
        rates = torch.where(
            self.first_losses > 0, values / self.first_losses, torch.ones_like(values)
        )
        target = gradients.mean().detach() * (rates / rates.mean()) ** GRADNORM_ALPHA
        return (gradients[1:] - target[1:]).abs().sum()

    def training_step(self, batch, batch_idx):
        embeddings = self.model.embedding(batch["ids"], batch["offsets"])
        # Against each class's soft target; dev scores stay against the true class, comparable across lam
        ce = self.loss_fn(self.model.linear(embeddings), self.targets[batch["target"]])
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
        extra = [values[name] for name in self.extra_losses]
        # The extra losses' weights: as given, or learned (GradNorm), as constants in the model's loss
        if self.hparams["gradnorm"]:
            weights = self.log_weights.exp().detach()
            gradnorm = self.gradnorm_loss(embeddings, [ce, *extra])
        else:
            weights = [self.hparams[f"{name}_weight"] for name in self.extra_losses]
            gradnorm = 0.0
        loss = ce + sum(weight * value for weight, value in zip(weights, extra))
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
            | {
                f"train_{name}_weight": weight
                for name, weight in zip(self.extra_losses, weights)
            },
            batch_size=n,
        )
        # train_loss is the model's objective; the probes and the log weights (GradNorm) train on their own losses
        optimizers, schedulers = self.optimizers(), self.lr_schedulers()
        # Lists: configure_optimizers returns two of each, three with GradNorm
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
        # GradNorm's log weights take their own optimizer
        own = [embedding] + ([self.log_weights] if self.hparams["gradnorm"] else [])
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
        if self.hparams["gradnorm"]:
            optimizers.append(torch.optim.Adam([self.log_weights], lr=GRADNORM_LR))
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=int(self.trainer.estimated_stepping_batches),
            )
            for optimizer in optimizers
        ]
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
