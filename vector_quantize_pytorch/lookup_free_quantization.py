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

from einops import rearrange, reduce, pack, unpack

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

# entropy

def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# convert to bit representations and back

def decimal_to_bits(x, bits):
    device = x.device

    x = x.int()

    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    x = rearrange(x, 'b n -> b n 1')

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, 'b n d -> b n d')
    return bits * 2 - 1

def bits_to_decimal(x, bits):
    device, dtype = x.device, x.dtype

    x = (x > 0).int()

    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)
    dec = reduce(x * mask, 'b n d -> b n', 'sum')
    return dec

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
        is_img_or_video = indices.ndim >= 3

        # rearrange if image or video into (batch, seq, dimension)

        if is_img_or_video:
            indices, ps = pack_one(indices, 'b *')

        # indices to codes, which are bits of either -1 or 1

        codes = decimal_to_bits(indices, self.dim)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = unpack_one(codes, ps, 'b * d')
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

        # quantize by eq 3.

        greater_than_zero = x > 0
        ones = torch.ones_like(x)

        quantized = torch.where(greater_than_zero, ones, -ones)

        # use straight-through gradients with tanh if training

        if self.training:
            x = torch.tanh(x * inv_temperature)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = bits_to_decimal(x, self.dim)

        # entropy aux loss (todo)

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

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b *')

        # bits to decimal for the codebook indices

        return Return(x, indices, entropy_aux_loss)
