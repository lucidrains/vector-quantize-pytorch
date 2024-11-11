# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vector_quantize_pytorch import LFQ, Sequential

lr = 3e-4
train_iter = 1000
seed = 1234
codebook_size = 2 ** 8
entropy_loss_weight = 0.02
diversity_gamma = 1.
spherical = True

device = "cuda" if torch.cuda.is_available() else "cpu"

def LFQAutoEncoder(
    codebook_size,
    **vq_kwargs
):
    assert log2(codebook_size).is_integer()
    quantize_dim = int(log2(codebook_size))

    return Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # In general norm layers are commonly used in Resnet-based encoder/decoders
        # explicitly add one here with affine=False to avoid introducing new parameters
        nn.GroupNorm(4, 32, affine=False),
        nn.Conv2d(32, quantize_dim, kernel_size=1),
        LFQ(dim=quantize_dim, **vq_kwargs),
        nn.Conv2d(quantize_dim, 32, kernel_size=3, stride=1, padding=1),
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
        out, indices, entropy_aux_loss = model(x)
        out = out.clamp(-1., 1.)

        rec_loss = F.l1_loss(out, x)
        (rec_loss + entropy_aux_loss).backward()

        opt.step()
        pbar.set_description(
              f"rec loss: {rec_loss.item():.3f} | "
            + f"entropy aux loss: {entropy_aux_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / codebook_size * 100:.3f}"
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

model = LFQAutoEncoder(
    codebook_size = codebook_size,
    entropy_loss_weight = entropy_loss_weight,
    diversity_gamma = diversity_gamma,
    spherical = spherical
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)

train(model, train_dataset, train_iterations=train_iter)
