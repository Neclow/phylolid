"""fastText's supervised model in PyTorch, as ConLID's cross-entropy baseline, its phylogenetic soft labels and its
Lightning training module."""

import torch

from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import MetricCollection

from src._config import PSEUDO_FAMILIES
from src.data.datasets import collate_features
from src.lid.metrics import KnownAccuracy, KnownMacroF1


class FastText(nn.Module):
    """Mean of a text's feature embeddings, then a linear layer over the classes."""

    def __init__(self, n_features, n_classes, dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_features, dim, mode="mean")
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, ids, offsets):
        return self.linear(self.embedding(ids, offsets))


def levels(a, b):
    """Glottolog levels between two lineages (family ... languoid): up from one to their lowest common ancestor, then
    down to the other (sisters: 2)."""
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    return len(a) + len(b) - 2 * common


def soft_targets(lineages, lam, beta):
    """Training target of each class (rows) over the classes: the class keeps 1 - lam, and lam goes to the other classes
    of its family, by exp(-beta * levels between them). One-hot (cross-entropy) when lam is 0, and for a class without
    relatives: an isolate, or a member of a pseudo-family."""
    targets = torch.eye(len(lineages))
    for y, lineage in enumerate(lineages):
        relatives = [
            c
            for c, other in enumerate(lineages)
            if c != y and other[0] == lineage[0] and lineage[0] not in PSEUDO_FAMILIES
        ]
        if lam > 0 and relatives:
            distances = torch.tensor(
                [float(levels(lineage, lineages[c])) for c in relatives]
            )
            targets[y, y] = 1 - lam
            targets[y, relatives] = lam * torch.softmax(-beta * distances, dim=0)
    return targets


class LitFastText(LightningModule):
    """Trains FastText with AdamW and a linearly decaying learning rate (ConLID's LIDCE), against soft_targets: lam
    0 is cross-entropy. The classes' lineages are needed for training only, to build the targets.
    """

    # Buffer of the training targets, typed as torchmetrics does
    targets: torch.Tensor

    def __init__(self, n_features, classes, dim, lr, lam=0.0, beta=0.5, lineages=None):
        super().__init__()
        # Saved with the checkpoint, classes (glottocodes in output order) included
        self.save_hyperparameters(ignore=["lineages"])
        if lineages is not None:
            self.register_buffer(
                "targets", soft_targets(lineages, lam, beta), persistent=False
            )
        self.model = FastText(n_features, len(classes), dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            [KnownAccuracy(len(classes)), KnownMacroF1(len(classes))],
            prefix="valid_",
        )

    def forward(self, ids, offsets):
        return self.model(ids, offsets)

    def training_step(self, batch, batch_idx):
        # Against each class's soft target; dev scores stay against the true class, comparable across lam
        logits = self(batch["ids"], batch["offsets"])
        loss = self.loss_fn(logits, self.targets[batch["target"]])
        # The number of texts (Lightning would infer it from the ids)
        self.log("train_loss", loss, batch_size=len(batch["target"]))
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
