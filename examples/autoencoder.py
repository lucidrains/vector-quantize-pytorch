# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from vector_quantize_pytorch import VectorQuantize

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

        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.quantizer = (
            VectorQuantize(
                dim=dim, codebook_size=codebook_size, accept_image_fmap=True
            ),
        )

    def forward(self, x):
        feat = self.encoder(x)
        quant, indices, commit_loss = self.quantizer(feat)
        out = self.decoder(quant)

        return out.clamp(-1, 1), indices, commit_loss

    def training_step(self, batch, batch_idx):
        out, indices, commit_loss = self.forward(batch)
        rec_loss = (out - batch).abs().mean()
        loss = rec_loss + self.alpha * commit_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_commit_loss", commit_loss, prog_bar=True)
        self.log(
            "train_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        out, indices, commit_loss = self.forward(batch)
        rec_loss = (out - batch).abs().mean()
        loss = rec_loss + self.alpha * commit_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
        self.log(
            "val_indices",
            indices.unique().numel() / self.codebook_size * 100,
            prog_bar=True,
        )


class LitFashionMNIST(LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = v2.Compose([v2.ToTensor(), v2.Normalize((0.5,), (0.5,))])

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, transform=self.transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)


model = SimpleVQAutoEncoder()
data = FashionMNIST()
trainer = Trainer(logger=True, max_epochs=100)

trainer.fit(model=model, datamodule=data)
