"""Module containing everything needed to train a simple VQ autoencoder.

This module contains the LightningModule subclass SimpleVQAutoEncoder,
which holds a basic implementations of a VQ VAE model using the VectorQuantize
module of the project.
Implementation inspired by https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

At the end of the module you can find a short script using the Trainer of Lightning to 
train the model for 10 epochs, using the FashionMNIST dataset as defined in 
the module data.py
"""
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn.functional import l1_loss

from examples.data import LitFashionMNIST
from vector_quantize_pytorch import VectorQuantize

seed_everything(1234, workers=True)


class SimpleVQAutoEncoder(LightningModule):
    """A simple VQ AutoEncoder using the classical VQ."""
    def __init__(
        self,
        dim: int = 32,
        codebook_size: int = 256,
        alpha: float = 10,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.alpha = alpha
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.quantizer = VectorQuantize(
            dim=dim, codebook_size=codebook_size, accept_image_fmap=True
        )

    def forward(self, x):
        feat = self.encoder(x)
        quant, indices, commit_loss = self.quantizer(feat)
        out = self.decoder(quant)

        return out.clamp(-1, 1), indices, commit_loss

    def training_step(self, batch, batch_idx):
        input, _ = batch
        out, indices, commit_loss = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss + self.alpha * commit_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_commit_loss", commit_loss, prog_bar=True)
        self.log(
            "train_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input, _ = batch
        out, indices, commit_loss = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss + self.alpha * commit_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
        self.log(
            "val_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = SimpleVQAutoEncoder()
data = LitFashionMNIST()
logger = TensorBoardLogger(save_dir=".", name="base_vqvae")
trainer = Trainer(
    logger=logger, max_epochs=10, callbacks=[RichProgressBar(), RichModelSummary()]
)

trainer.fit(model=model, datamodule=data)
