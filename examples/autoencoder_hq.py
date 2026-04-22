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

try:
    import fire
except ImportError:  # pragma: no cover
    fire = None
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vector_quantize_pytorch import HierarchicalVQ, Sequential


def count_active_codes(indices):
    flat = [ind.reshape(-1) for ind in indices]
    return torch.cat(flat).unique().numel()


def SimpleHQAutoEncoder(
    dim = 32,
    scales = (1, 2, 4, 7),
    quant_resi = 0.5,
    share_quant_resi = 1,
    rotation_trick = False,
    **hq_kwargs
):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.GELU(),
        nn.Conv2d(16, dim, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        HierarchicalVQ(
            dim = dim,
            accept_image_fmap = True,
            scales = scales,
            quant_resi = quant_resi,
            share_quant_resi = share_quant_resi,
            rotation_trick = rotation_trick,
            **hq_kwargs
        ),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(dim, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.GELU(),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1),
    )


def train(
    train_iter = 1000,
    lr = 3e-4,
    dim = 32,
    num_codes = 512,
    seed = 1234,
    scales = (1, 2, 4, 7),
    quant_resi = 0.5,
    share_quant_resi = 1,
    rotation_trick = False,
    alpha = 10.,
    batch_size = 256
):
    torch.random.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleHQAutoEncoder(
        dim = dim,
        scales = scales,
        quant_resi = quant_resi,
        share_quant_resi = share_quant_resi,
        rotation_trick = rotation_trick,
        codebook_size = num_codes,
        kmeans_init = True,
        kmeans_iters = 10,
    ).to(device)

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

        out, indices, cmt_loss = model(x)
        out = out.clamp(-1., 1.)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()
        opt.step()

        active_percent = count_active_codes(indices) / num_codes * 100
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            f"cmt loss: {cmt_loss.item():.3f} | "
            f"active %: {active_percent:.3f}"
        )


if __name__ == "__main__":
    if fire is None:
        raise ImportError('fire is required to use the CLI entrypoint for this example')
    fire.Fire(train)
