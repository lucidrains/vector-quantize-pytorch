# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


class LitFashionMNIST(LightningDataModule):
    def __init__(self, data_dir: str = ".", batch_size: int = 256):
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
