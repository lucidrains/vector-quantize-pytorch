# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from torch import nn
from torch.nn.functional import l1_loss

from vector_quantize_pytorch import VectorQuantize

from .data import LitFashionMNIST

seed_everything(1234, workers=True)


class SimpleVQAutoEncoder(LightningModule):
    def __init__(
        self,
        dim: int = 32,
        codebook_size: int = 256,
        alpha: float = 10,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.dim = dim
        self.codbook_size = codebook_size
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = SimpleVQAutoEncoder()
data = LitFashionMNIST()
trainer = Trainer(
    logger=True, max_epochs=100, callbacks=[RichProgressBar(), RichModelSummary()]
)

trainer.fit(model=model, datamodule=data)
