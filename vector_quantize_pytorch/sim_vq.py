from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import einsum, rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

# class

class SimVQ(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        init_fn: Callable = identity,
        accept_image_fmap = False
    ):
        super().__init__()
        self.accept_image_fmap = accept_image_fmap

        codebook = torch.randn(codebook_size, dim) * (dim ** -0.5)
        codebook = init_fn(codebook)

        # the codebook is actually implicit from a linear layer from frozen gaussian or uniform

        self.codebook_to_codes = nn.Linear(dim, dim, bias = False)
        self.register_buffer('codebook', codebook)

    def forward(
        self,
        x
    ):
        if self.accept_image_fmap:
            x = rearrange(x, 'b d h w -> b h w d')
            x, inverse_pack = pack_one(x, 'b * d')

        implicit_codebook = self.codebook_to_codes(self.codebook)

        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)

        # select codes

        quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)

        # commit loss

        commit_loss = (
            0.25 * F.mse_loss(x, quantized.detach()) +
            F.mse_loss(x.detach(), quantized)
        )

        quantized = (quantized - x).detach() + x

        if self.accept_image_fmap:
            quantized = inverse_pack(quantized)
            quantized = rearrange(quantized, 'b h w d-> b d h w')

            indices = inverse_pack(indices, 'b *')

        return quantized, indices, commit_loss

# main

if __name__ == '__main__':

    x = torch.randn(1, 512, 32, 32)

    sim_vq = SimVQ(
        dim = 512,
        codebook_size = 1024,
        accept_image_fmap = True
    )

    quantized, indices, commit_loss = sim_vq(x)

    assert x.shape == quantized.shape
