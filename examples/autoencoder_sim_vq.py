# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vector_quantize_pytorch import SimVQ, Sequential

lr = 3e-4
train_iter = 10000
num_codes = 256
seed = 1234

rotation_trick = True  # rotation trick instead ot straight-through
use_mlp = True         # use a one layer mlp with relu instead of linear

device = "cuda" if torch.cuda.is_available() else "cpu"

def SimVQAutoEncoder(**vq_kwargs):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        SimVQ(dim=32, channel_first = True, **vq_kwargs),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
    )

def train(model, train_loader, train_iterations=1000, alpha=10):
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

        out, indices, cmt_loss = model(x)
        out = out.clamp(-1., 1.)

        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()

        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
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

model = SimVQAutoEncoder(
    codebook_size = num_codes,
    rotation_trick = rotation_trick,
    codebook_transform = nn.Sequential(
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    ) if use_mlp else None
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
