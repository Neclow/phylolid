"""fastText's supervised model in PyTorch, as ConLID's cross-entropy baseline, its phylogenetic losses and its
Lightning training module."""

import torch

from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import MetricCollection

from src.data.datasets import collate_features
from src.lid.metrics import KnownAccuracy, KnownMacroF1
from src.lid.tree import tree_distances
from src.lid.tree_losses import TreeLosses


class FastText(nn.Module):
    """Mean of a text's feature embeddings, then a linear layer over the classes."""

    def __init__(self, n_features, n_classes, dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_features, dim, mode="mean")
        self.linear = nn.Linear(dim, n_classes)

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


class LitFastText(LightningModule):
    """Trains FastText with AdamW and a linearly decaying learning rate (ConLID's LIDCE), against soft_targets (lam
    0: cross-entropy), plus tree_weight and quartet_weight times the TreeLosses of the sentence embeddings. The
    classes' lineages are needed for training only.
    """

    # Buffer of the training targets, typed as torchmetrics does, and the tree losses (training only)
    targets: torch.Tensor
    tree_losses: TreeLosses

    def __init__(
        self,
        n_features,
        classes,
        dim,
        lr,
        lam=0.0,
        beta=0.5,
        tree_weight=0.0,
        quartet_weight=0.0,
        lineages=None,
    ):
        super().__init__()
        # Saved with the checkpoint, classes (glottocodes in output order) included
        self.save_hyperparameters(ignore=["lineages"])
        if lineages is not None:
            self.register_buffer(
                "targets", soft_targets(lineages, lam, beta), persistent=False
            )
            self.tree_losses = TreeLosses(tree_distances(lineages), dim)
        self.model = FastText(n_features, len(classes), dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            [KnownAccuracy(len(classes)), KnownMacroF1(len(classes))],
            prefix="valid_",
        )

    def forward(self, ids, offsets):
        return self.model(ids, offsets)

    def training_step(self, batch, batch_idx):
        embeddings = self.model.embedding(batch["ids"], batch["offsets"])
        # Against each class's soft target; dev scores stay against the true class, comparable across lam
        ce = self.loss_fn(self.model.linear(embeddings), self.targets[batch["target"]])
        weights = {
            "tree": self.hparams["tree_weight"],
            "quartet": self.hparams["quartet_weight"],
        }
        # Always computed, as a measure of how tree-like the embeddings are; with a gradient only when weighted
        with torch.set_grad_enabled(any(weights.values())):
            tree, quartet = self.tree_losses(embeddings, batch["target"])
        loss = ce
        for name, value in {"tree": tree, "quartet": quartet}.items():
            if weights[name]:
                loss = loss + weights[name] * value
        # The number of texts (Lightning would infer it from the ids)
        n = len(batch["target"])
        self.log_dict(
            {
                "train_loss": loss,
                "train_ce_loss": ce,
                "train_tree_loss": tree,
                "train_quartet_loss": quartet,
            },
            batch_size=n,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["ids"], batch["offsets"])
        loss = self.loss_fn(logits, batch["target"])
        self.log("valid_loss", loss, batch_size=len(batch["target"]))
        # Metric objects, so that Lightning computes them over the whole dev split
        self.valid_metrics.update(logits, batch["target"])
        self.log_dict(self.valid_metrics, batch_size=len(batch["target"]))

    def configure_optimizers(self):
        # ConLID's settings (appendix B.2): no weight decay, the learning rate falls linearly to 0 over training
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


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
