"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

basically a 2-level FSQ (Finite Scalar Quantization) with entropy loss
https://arxiv.org/abs/2309.15505
"""

from math import log2
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        diversity_gamma = 1.
    ):
        super().__init__()

        # some asesrt validations

        assert exists(dim) or exists(codebook_size)
        assert not exists(codebook_size) or log2(codebook_size).is_integer()

        codebook_size = default(codebook_size, 2 ** dim)
        dim = default(dim, int(log2(codebook_size)))

        assert (2 ** dim) == codebook_size, f'2 ^ dimension ({dim}) must be equal to the codebook size ({codebook_size})'

        self.dim = dim

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer('zero', torch.zeros(1,), persistent = False)

    def indices_to_codes(self, indices):
        raise NotImplementedError

    def forward(self, x):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        """

        is_img_or_video = x.ndim >= 4

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

        raise NotImplementedError
