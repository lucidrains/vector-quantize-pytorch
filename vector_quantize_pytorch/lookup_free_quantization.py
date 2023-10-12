"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

basically a 2-level FSQ (Finite Scalar Quantization) with entropy loss
https://arxiv.org/abs/2309.15505
"""

from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module

from einops import rearrange, reduce, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# entropy

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        diversity_gamma = 2.5
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        dim = default(dim, codebook_dim)

        self.project_in = nn.Linear(dim, codebook_dim) if dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if dim != codebook_dim else nn.Identity()

        self.dim = dim
        self.codebook_dim = codebook_dim

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.zeros(1,), persistent = False)

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        is_img_or_video = indices.ndim >= 3

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).float()
        codes = bits * 2 - 1

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(
        self,
        x,
        inv_temperature = 1.
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        """

        is_img_or_video = x.ndim >= 4

        # rearrange if image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim

        x = self.project_in(x)

        # quantize by eq 3.

        ones = torch.ones_like(x)
        quantized = torch.where(x > 0, ones, -ones)

        # use straight-through gradients with tanh if training

        if self.training:
            x = torch.tanh(x * inv_temperature)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), 'b n d -> b n', 'sum')

        # entropy aux loss

        if self.training:
            prob = (x * inv_temperature).sigmoid()

            bit_entropy = binary_entropy(prob).mean()

            avg_prob = reduce(prob, 'b n d -> b d', 'mean')
            codebook_entropy = binary_entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each bit, so each scalar commits to one latent binary bit or the other
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used

            entropy_aux_loss = bit_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = self.zero

        entropy_aux_loss = entropy_aux_loss * self.entropy_loss_weight

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b *')

        # bits to decimal for the codebook indices

        return Return(x, indices, entropy_aux_loss)
