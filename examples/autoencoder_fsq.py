#!/usr/bin/env uv run
# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "tqdm",
#   "fire",
#   "einops",
#   "einx",
# ]
# ///

from tqdm.auto import trange

import math
import fire
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW

from vector_quantize_pytorch import FSQ, Sequential

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

def SimpleFSQAutoEncoder(levels: list[int]):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(32, len(levels), kernel_size = 1),
        FSQ(levels),
        nn.Conv2d(len(levels), 32, kernel_size = 3, stride = 1, padding = 1),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.GELU(),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1),
    )

def train(
    train_iter = 1000,
    lr = 3e-4,
    levels = [8, 6, 5],
    seed = 1234,
    batch_size = 256
):
    torch.random.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_codes = math.prod(levels)

    model = SimpleFSQAutoEncoder(levels).to(device)

    opt = AdamW(model.parameters(), lr = lr)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = DataLoader(
        datasets.FashionMNIST(root = "~/data/fashion_mnist", train = True, download = True, transform = transform),
        batch_size = batch_size,
        shuffle = True,
    )

    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    dl_iter = iterate_dataset(train_dataset)

    pbar = trange(train_iter)

    for _ in pbar:
        opt.zero_grad()
        x, _ = next(dl_iter)

        out, indices = model(x)
        out = out.clamp(-1., 1.)

        rec_loss = (out - x).abs().mean()
        rec_loss.backward()

        opt.step()

        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

if __name__ == "__main__":
    fire.Fire(train)
