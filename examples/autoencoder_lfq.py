# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from math import log2

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from torch import nn
from torch.nn.functional import l1_loss

from vector_quantize_pytorch import LFQ

from .data import LitFashionMNIST

seed_everything(1234, workers=True)


class LFQAutoEncoder(LightningModule):
    def __init__(
        self,
        codebook_size: int = 2**8,
        diversity_gamma: float = 1.0,
        entropy_loss_weight: float = 0.02,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.lr = lr

        self.quantize_dim = int(log2(self.codebook_size))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # In general norm layers are commonly used in Resnet-based encoder/decoders
            # explicitly add one here with affine=False to avoid introducing new parameters
            nn.GroupNorm(4, 32, affine=False),
            nn.Conv2d(32, self.quantize_dim, kernel_size=1),
        )

        self.quantizer = LFQ(
            dim=self.quantize_dim,
            diversity_gamma=self.diversity_gamma,
            entropy_loss_weight=self.entropy_loss_weight,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.quantize_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x, indices, entropy_aux_loss = self.quantizer(x)
        x = self.decoder(x)
        return x.clamp(-1, 1), indices, entropy_aux_loss

    def training_step(self, batch, batch_idx):
        input, _ = batch
        out, indices, entropy_aux_loss = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss + entropy_aux_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_entropy_aux_loss", entropy_aux_loss, prog_bar=True)
        self.log(
            "train_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        input, _ = batch
        out, indices, entropy_aux_loss = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss + entropy_aux_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_entropy_aux_loss", entropy_aux_loss, prog_bar=True)
        self.log(
            "val_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = LFQAutoEncoder()
data = LitFashionMNIST()
trainer = Trainer(
    logger=True, max_epochs=100, callbacks=[RichProgressBar(), RichModelSummary()]
)

trainer.fit(model=model, datamodule=data)
