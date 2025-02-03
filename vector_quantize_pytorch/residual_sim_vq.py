from __future__ import annotations

import random
from math import ceil
from functools import partial, cache
from itertools import zip_longest

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.distributed as dist

from vector_quantize_pytorch.sim_vq import SimVQ, pack_one

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# main class

class ResidualSimVQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_size,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        channel_first = False,
        rotation_trick = True, # rotation trick from @cfifty, on top of sim vq
        **sim_vq_kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'

        self.channel_first = channel_first

        self.num_quantizers = num_quantizers

        # define sim vq across layers

        self.layers = ModuleList([SimVQ(dim = dim, codebook_size = codebook_size, rotation_trick = rotation_trick, channel_first = channel_first, **sim_vq_kwargs) for _ in range(num_quantizers)])

        # quantize dropout

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebook_size(self):
        return first(self.layers).codebook_size
    
    @property
    def codebook_dim(self):
        return first(self.layers).codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (images)

        indices, inverse = pack_one(indices, 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # if (channel_first = True) then return shape (quantize, batch, height, width, dimension)

        all_codes = inverse(all_codes, 'q b * d')

        if self.channel_first:
            all_codes = rearrange(all_codes, 'q b ... d -> q b d ...')

        return all_codes

    def get_output_from_indices(self, indices):
        all_codes = self.get_codes_from_indices(indices)
        summed_residual_codes = reduce(all_codes, 'q ... -> ...', 'sum')
        return summed_residual_codes

    def forward(
        self,
        x,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[2:]) if self.channel_first else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            null_loss = torch.full((), 0., device = device, dtype = x.dtype)

        # save all inputs across layers, for use during expiration at end under shared codebook setting

        all_residuals = []

        # go through the layers

        for quantizer_index, sim_vq in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            # save for expiration

            all_residuals.append(residual)

            # sim vq forward

            quantized, *rest = sim_vq(residual)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # stack all losses and indices

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)
