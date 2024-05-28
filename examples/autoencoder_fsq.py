# FashionMnist VQ experiment with various settings, using FSQ.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py


import math

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn.functional import l1_loss

from examples.data import LitFashionMNIST
from vector_quantize_pytorch import FSQ

seed_everything(1234, workers=True)


class SimpleFSQAutoEncoder(LightningModule):
    def __init__(self, levels: list[int] = [8, 6, 5], lr: float = 3e-4):
        super().__init__()
        self.levels = levels
        self.lr = lr
        self.num_codes = math.prod(levels)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, len(levels), kernel_size=1),
        )

        self.quantizer = FSQ(self.levels)

        self.decoder = nn.Sequential(
            nn.Conv2d(len(levels), 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x, indices = self.quantizer(x)
        x = self.decoder(x)
        return x.clamp(-1, 1), indices

    def training_step(self, batch, batch_idx):
        input, _ = batch
        out, indices = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_indices",
            indices.unique().numel() / self.num_codes * 100,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        input, _ = batch
        out, indices = self.forward(input)
        rec_loss = l1_loss(out, input)
        loss = rec_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_indices",
            indices.unique().numel() / self.num_codes * 100,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = SimpleFSQAutoEncoder()

data = LitFashionMNIST()
logger = TensorBoardLogger(save_dir=".", name="FSQ")
trainer = Trainer(
    logger=True, max_epochs=10, callbacks=[RichProgressBar(), RichModelSummary()]
)

trainer.fit(model=model, datamodule=data)
