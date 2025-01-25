# FashionMnist VQ experiment with various settings, using FSQ.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vector_quantize_pytorch import FSQ, Sequential


lr = 3e-4
train_iter = 1000
levels = [8, 6, 5] # target size 2^8, actual size 240
num_codes = math.prod(levels)
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


def SimpleFSQAutoEncoder(levels: list[int]):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, len(levels), kernel_size=1),
        FSQ(levels),
        nn.Conv2d(len(levels), 32, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
    )


def train(model, train_loader, train_iterations=1000):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices = model(x)
        out = out.clamp(-1., 1.)

        rec_loss = (out - x).abs().mean()
        rec_loss.backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

torch.random.manual_seed(seed)
model = SimpleFSQAutoEncoder(levels).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
