#!/usr/bin/env uv run
# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "tqdm",
#   "fire",
#   "einops",
#   "einx",
#   "x-transformers==2.16.1",
# ]
# ///

from tqdm.auto import trange

import fire

import torch
import torch.nn as nn
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

from einops import rearrange
from einops.layers.torch import Rearrange

from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize, Sequential

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

def VQBridgeViT(
    dim,
    depth,
    input_dim = None,
    patch_size = 1,
    dim_head = 64,
    heads = 4,
    num_registers = 0
):
    # Credit goes to Mahdi (@mahdip72) for his experiments that found the best
    # set of hyperparameters for the ViT used in FVQ, which is patch_size 1 (becomes a regular transformer encoder) and critically, having register tokens (we will do 2 here)
    # see experiments at https://github.com/lucidrains/vector-quantize-pytorch/issues/239#issuecomment-3888240360

    input_dim = default(input_dim, dim)

    project_in_out_kwargs = dict()

    inner_dim = input_dim * patch_size

    if patch_size > 1 or inner_dim != dim:
        project_in_out_kwargs.update(
            project_in = nn.Sequential(
                Rearrange('b (n p) c -> b n (p c)', p = patch_size),
                nn.Linear(inner_dim, dim, bias = False)
            ),
            project_out = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = False),
                Rearrange('b n (p c) -> b (n p) c', p = patch_size)
            )
        )

    return ContinuousTransformerWrapper(
        num_memory_tokens = num_registers,
        attn_layers = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            pre_norm_has_final_norm = False
        ),
        **project_in_out_kwargs
    )

def SimpleVQAutoEncoder(
    dim = 32,
    vq_bridge: Module | None = None,
    rotation_trick = True,
    **vq_kwargs
):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        VectorQuantize(
            dim = dim,
            accept_image_fmap = True,
            rotation_trick = rotation_trick,
            vq_bridge = vq_bridge,
            **vq_kwargs
        ),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.GELU(),
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1),
    )

def train(
    train_iter = 1000,
    lr = 3e-4,
    dim = 32,
    num_codes = 256,
    seed = 1234,
    patch_size = 1,
    no_bridge = False,
    rotation_trick = False,
    num_registers = 2,
    heads = 4,
    vq_dim = 256,
    vq_depth = 1,
    alpha = 10,
    batch_size = 256
):
    torch.random.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq_bridge = None

    if not no_bridge:
        vq_bridge = VQBridgeViT(
            dim = vq_dim,
            depth = vq_depth,
            input_dim = dim,
            patch_size = patch_size,
            heads = heads,
            num_registers = num_registers
        )

    model = SimpleVQAutoEncoder(
        dim = dim,
        vq_bridge = vq_bridge,
        rotation_trick = rotation_trick
        codebook_size = num_codes,
        learnable_codebook = True,
        in_place_codebook_optimizer = lambda *args, **kwargs: SGD(*args, **kwargs, lr = 1e-3),
        ema_update = False,
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

        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            f"cmt loss: {cmt_loss.item():.3f} | "
            f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

if __name__ == "__main__":
    fire.Fire(train)
