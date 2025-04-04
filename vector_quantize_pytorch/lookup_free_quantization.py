"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear

class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = False,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True               # will force the quantization step to be full precision
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)

        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = self.maybe_l2norm(codes)

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(
        self,
        x,
        inv_temperature = 100.,
        return_loss_breakdown = False,
        mask = None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # maybe soft clamp

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # maybe l2norm

        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices

            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

            # maybe l2norm

            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss

            if self.training:

                codebook = self.codebook

                if force_f32:
                    codebook = codebook.float()

                codebook = self.maybe_l2norm(codebook)

                # whether to only use a fraction of probs, for reducing memory

                input_for_entropy = original_input

                if exists(mask):
                    input_for_entropy = original_input[mask]

                input_for_entropy = rearrange(input_for_entropy, 'b n ... -> (b n) ...')

                if self.frac_per_sample_entropy < 1.:
                    # account for mask

                    num_tokens = input_for_entropy.size(0)
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens

                    sampled_input = input_for_entropy[rand_mask]

                    sampled_distance = -2 * einsum('... i d, j d -> ... i j', sampled_input, codebook)

                    sampled_prob = (-sampled_distance * inv_temperature).softmax(dim = -1)

                    per_sample_probs = sampled_prob
                else:

                    # the same as euclidean distance up to a constant
                    distance = -2 * einsum('... i d, j d -> ... i j', input_for_entropy, codebook)

                    prob = (-distance * inv_temperature).softmax(dim = -1)

                    per_sample_probs = prob

                # calculate per sample entropy

                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')

                avg_prob = maybe_distributed_mean(avg_prob)

                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            # commit loss

            if self.training and self.commitment_loss_weight > 0.:

                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction = 'none')

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim

        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # complete aux loss

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        # returns

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)
