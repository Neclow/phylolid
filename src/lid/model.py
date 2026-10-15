"""fastText's supervised model in PyTorch, as ConLID's cross-entropy baseline, and its Lightning training module."""

import torch

from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import MetricCollection

from src.lid.metrics import KnownAccuracy, KnownMacroF1


class FastText(nn.Module):
    """Mean of a text's feature embeddings, then a linear layer over the classes."""

    def __init__(self, n_features, n_classes, dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_features, dim, mode="mean")
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, ids, offsets):
        return self.linear(self.embedding(ids, offsets))


class LitFastText(LightningModule):
    """Trains FastText with cross-entropy, AdamW and a linearly decaying learning rate (ConLID's LIDCE)."""

    def __init__(self, n_features, classes, dim, lr):
        super().__init__()
        # Saved with the checkpoint, classes (glottocodes in output order) included
        self.save_hyperparameters()
        self.model = FastText(n_features, len(classes), dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            [KnownAccuracy(len(classes)), KnownMacroF1(len(classes))],
            prefix="valid_",
        )

    def forward(self, ids, offsets):
        return self.model(ids, offsets)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch["ids"], batch["offsets"]), batch["target"])
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
