from __future__ import annotations

import random
from math import ceil
from functools import partial, cache
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, arange, cat
from torch.nn import Module, ModuleList

import torch.distributed as dist
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, reduce, pack, unpack

import einx
from einx import get_at

# helper functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def unique(arr):
    return list({*arr})

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def frac_gradient(t, frac):
    detached = t.detach()

    if frac <= 0:
        return detached

    return frac * t + (1. - frac) * detached

# tensor helpers

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return first(unpack(out, packed_shape, inv_pattern))

    return packed, inverse

def batch_select(t, indices, pattern = None):

    if exists(pattern):
        indices = rearrange(indices, '... k -> (...) k')
        t, inv_pack = pack_one(t, pattern)


    batch_indices = arange(t.shape[0], device = t.device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    out = t[batch_indices, indices]

    if exists(pattern):
        out = inv_pack(out)

    return out

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# the mlp for generating the neural implicit codebook
# from Huijben et al. https://arxiv.org/abs/2401.14732

class MLP(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 4,             # they used 4 layers in the paper
        l2norm_output = False
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)

        self.proj_in = nn.Linear(2 * dim, dim)

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, dim)
            ))

        self.layers = layers
        self.l2norm_output = l2norm_output

    def forward(
        self,
        codes,
        *,
        condition
    ):
        one_headed = codes.ndim == 2

        condition, _ = pack([condition], 'b * d') # handle condition with ndim 2 - (batch, dim)

        if one_headed:
            codes = rearrange(codes, 'c d -> 1 c d')

        heads, num_codes, batch, seq_len = codes.shape[0], codes.shape[-2], condition.shape[0], condition.shape[-2]

        codes = repeat(codes, 'h c d -> h b n c d', n = seq_len, b = batch)
        condition = repeat(condition, 'b n d -> h b n c d', c = num_codes, h = heads)

        x = torch.cat((condition, codes), dim = -1)
        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        if self.l2norm_output:
            x = F.normalize(x, dim = -1)

        if not one_headed:
            return x

        return rearrange(x, '1 ... -> ...')

# main class

class ResidualVQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        dim,
        num_quantizers: int | None = None,
        codebook_size: int | tuple[int, ...],
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        implicit_neural_codebook = False, # QINCo from https://arxiv.org/abs/2401.14732
        mlp_kwargs: dict = dict(),
        beam_size = None,
        eval_beam_size = None,
        beam_score_quantizer_weights: list[float] | None = None,
        quant_grad_frac = 0.,
        **vq_kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        assert exists(num_quantizers) or isinstance(codebook_size, tuple)

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.accept_image_fmap = accept_image_fmap

        self.implicit_neural_codebook = implicit_neural_codebook

        if implicit_neural_codebook:
            vq_kwargs.update(
                learnable_codebook = True,
                ema_update = False
            )

        if shared_codebook:
            vq_kwargs.update(
                manual_ema_update = True,
                manual_in_place_optimizer_update = True
            )

        # take care of maybe different codebook sizes across depth

        codebook_sizes = cast_tuple(codebook_size, num_quantizers)

        num_quantizers = default(num_quantizers, len(codebook_sizes))
        assert len(codebook_sizes) == num_quantizers

        self.num_quantizers = num_quantizers

        self.codebook_sizes = codebook_sizes
        self.uniform_codebook_size = len(unique(codebook_sizes)) == 1

        # define vq across layers

        self.layers = ModuleList([VectorQuantize(dim = codebook_dim, codebook_size = layer_codebook_size, codebook_dim = codebook_dim, accept_image_fmap = accept_image_fmap, **vq_kwargs) for layer_codebook_size in codebook_sizes])

        assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        # determine whether is using ema update

        self.vq_is_ema_updating = first(self.layers).ema_update

        # gradient related - how much gradients to allow up the residual path (previous layers influence the layer above)

        self.quant_grad_frac = quant_grad_frac

        # beam size

        assert not (exists(eval_beam_size) and not exists(beam_size))

        self.beam_size = beam_size
        self.eval_beam_size = default(eval_beam_size, beam_size)

        # able to assign a different weight for the scoring at each quantizer layer

        beam_score_quantizer_weights = default(beam_score_quantizer_weights, [1.] * num_quantizers)
        assert len(beam_score_quantizer_weights) == num_quantizers

        self.register_buffer('beam_score_weights', tensor(beam_score_quantizer_weights), persistent = False)

        # setting up the MLPs for implicit neural codebooks
        
        self.mlps = None

        if implicit_neural_codebook:
            self.mlps = ModuleList([MLP(dim = codebook_dim, l2norm_output = first(self.layers).use_cosine_sim, **mlp_kwargs) for _ in range(num_quantizers - 1)])
        else:
            self.mlps = (None,) * (num_quantizers - 1)

        # sharing codebook logic

        self.shared_codebook = shared_codebook

        if not shared_codebook:
            return

        assert self.uniform_codebook_size

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook
    
    @property
    def codebook_size(self):
        return self.layers[0].codebook_size
    
    @property
    def codebook_dim(self):
        return self.layers[0].codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]

        codebooks = tuple(rearrange(codebook, '1 ... -> ...') for codebook in codebooks)

        if not self.uniform_codebook_size:
            return codebooks

        codebooks = torch.stack(codebooks)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        if not self.implicit_neural_codebook and self.uniform_codebook_size:

            all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        else:
            # else if using implicit neural codebook, or non uniform codebook sizes, codes will need to be derived layer by layer

            code_transform_mlps = (None, *self.mlps)

            all_codes = []
            quantized_out = 0.

            for codes, indices, maybe_transform_mlp in zip(self.codebooks, indices.unbind(dim = -1), code_transform_mlps):

                if exists(maybe_transform_mlp):
                    codes = maybe_transform_mlp(codes, condition = quantized_out)
                    layer_codes = get_at('b n [c] d, b n -> b n d', codes, indices)
                else:
                    layer_codes = get_at('[c] d, b n -> b n d', codes, indices)

                all_codes.append(layer_codes)
                quantized_out += layer_codes

            all_codes = torch.stack(all_codes)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        mask = None,
        indices: Tensor | list[Tensor] | None = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        beam_size = None,
        rand_quantize_dropout_fixed_seed = None
    ):

        # variables

        input_shape, num_quant, quant_dropout_multiple_of, return_loss, device = x.shape, self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        beam_size = default(beam_size, self.beam_size if self.training else self.eval_beam_size)

        is_beam_search = exists(beam_size) and beam_size > 1

        # projecting in

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = torch.zeros_like(x)
        residual = x

        all_losses = torch.empty((0,), device = device, dtype = torch.float32)
        all_indices = torch.empty((*input_shape[:-1], 0), device = device, dtype = torch.long)

        if isinstance(indices, list):
            indices = torch.stack(indices)

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

        # save all inputs across layers, for use during expiration at end under shared codebook setting, or ema update during beam search

        all_residuals = torch.empty((*input_shape[:-1], 0, input_shape[-1]), dtype = residual.dtype, device = device)

        # maybe prepare beam search

        if is_beam_search:
            prec_dims = x.shape[:-1]

            search_scores = torch.zeros((*prec_dims, 1), device = device, dtype = x.dtype)

            residual = rearrange(residual, '... d -> ... 1 d')
            quantized_out = rearrange(quantized_out, '... d -> ... 1 d')

            all_residuals = rearrange(all_residuals, '... l d -> ... 1 l d')
            all_indices = rearrange(all_indices, '... l -> ... 1 l')

            all_losses = all_losses.reshape(*input_shape[:-1], 1, 0)

        # setup the mlps for implicit neural codebook

        maybe_code_transforms = (None,) * len(self.layers)

        if self.implicit_neural_codebook:
            maybe_code_transforms = (None, *self.mlps)

        # go through the layers

        for quantizer_index, (vq, maybe_mlp) in enumerate(zip(self.layers, maybe_code_transforms)):

            is_last_step = (quantizer_index == (len(self.layers) - 1)) if not should_quantize_dropout else quantizer_index == rand_quantize_dropout_index

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices = pad_at_dim(all_indices, (0, 1), value = -1, dim = -1)
                all_losses = pad_at_dim(all_losses, (0, 1), value = 0, dim = -1)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            # setup the transform code function to be passed into VectorQuantize forward

            if exists(maybe_mlp):
                maybe_mlp = partial(maybe_mlp, condition = quantized_out)

            # save the residual input for maybe expiration as well as ema update after beam search

            all_residuals = cat((all_residuals, rearrange(residual, '... d -> ... 1 d')), dim = -2)

            # vector quantize forward

            quantized, *rest = vq(
                residual,
                mask = mask,
                indices = layer_indices,
                sample_codebook_temp = sample_codebook_temp,
                freeze_codebook = freeze_codebook,
                codebook_transform_fn = maybe_mlp,
                topk = beam_size if is_beam_search else None
            )

            # cross entropy loss for some old paper

            if return_loss:
                ce_loss = first(rest)
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            # handle expanding first residual if doing beam search

            if is_beam_search:

                score_weight = self.beam_score_weights[quantizer_index]
                search_scores = einx.add('... j, ... j k -> ... (j k)', search_scores, -loss * score_weight)

                residual = rearrange(residual, '... j d -> ... j 1 d')
                quantized_out = rearrange(quantized_out, '... j d -> ... j 1 d')

                all_residuals = repeat(all_residuals, '... j l d -> ... (j k) l d', k = beam_size)

            # core residual vq logic

            residual = residual - frac_gradient(quantized, self.quant_grad_frac)
            quantized_out = quantized_out + quantized

            # handle sort and topk beams

            if is_beam_search:
                residual = rearrange(residual, '... j k d -> ... (j k) d')
                quantized_out = rearrange(quantized_out, '... j k d -> ... (j k) d')

                # broadcat the indices

                all_indices = repeat(all_indices, '... j l -> ... j k l', k = embed_indices.shape[-1])
                embed_indices = rearrange(embed_indices, '... j k -> ... j k 1')

                all_indices = cat((all_indices, embed_indices), dim = -1)
                all_indices = rearrange(all_indices, '... j k l -> ... (j k) l')

                # broadcat the losses

                all_losses = repeat(all_losses, '... j l -> ... j k l', k = loss.shape[-1])
                loss = rearrange(loss, '... -> ... 1')

                all_losses = cat((all_losses, loss), dim = -1)
                all_losses = rearrange(all_losses, '... j k l -> ... (j k) l')

                # handle sort and selection of highest beam size

                layer_beam_size = beam_size if not is_last_step else 1

                if search_scores.shape[-1] > layer_beam_size:
                    search_scores, select_indices = search_scores.topk(layer_beam_size, dim = -1)

                    residual = batch_select(residual, select_indices, '* k d')
                    quantized_out = batch_select(quantized_out, select_indices, '* k d')

                    all_indices = batch_select(all_indices, select_indices, '* k l')
                    all_losses = batch_select(all_losses, select_indices, '* k l')

                    all_residuals = batch_select(all_residuals, select_indices, '* k l d')
            else:
                # aggregate indices and losses

                all_indices = cat((all_indices, rearrange(embed_indices, '... -> ... 1')), dim = -1)

                all_losses = cat((all_losses, rearrange(loss, '... -> ... 1')), dim = -1)

        # handle beam search

        if is_beam_search:

            quantized_out, all_indices, all_losses, all_residuals = [t[..., 0, :] for t in (quantized_out, all_indices, all_losses, all_residuals)]

            # handle commit loss, which should be the average

            if exists(mask):
                all_losses = einx.where('..., ... l,', mask, all_losses, 0.)
                all_losses = reduce(all_losses, '... l -> l', 'sum') / mask.sum(dim = -1).clamp_min(1e-4)
            else:
                all_losses = reduce(all_losses, '... l -> l', 'mean')

            # handle updating ema

            if self.training:
                for vq, layer_input, indices in zip(self.layers, all_residuals.unbind(dim = -2), all_indices.unbind(dim = -1)): # in the case of quantize dropout, zip will terminate with the shorter sequence, which should be all_residuals

                    if self.vq_is_ema_updating:
                        vq.update_ema_indices(layer_input, indices, mask = mask)

                    batch_samples = layer_input[mask] if exists(mask) else layer_input
                    vq.expire_codes_(batch_samples)

        # if shared codebook, update ema only at end

        if self.training and self.shared_codebook:

            shared_layer = first(self.layers)
            shared_layer._codebook.update_ema()
            shared_layer.update_in_place_optimizer()

            all_codes_for_expire = rearrange(all_residuals, '... n l d -> ... (n l) d')
            shared_layer.expire_codes_(all_codes_for_expire)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret

# grouped residual vq

class GroupedResidualVQ(Module):
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualVQ(
                dim = dim_per_group,
                accept_image_fmap = accept_image_fmap,
                **kwargs
            ))

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    def forward(
        self,
        x,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        mask = None,
    ):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim = split_dim)

        indices = default(indices, tuple())
        return_ce_loss = len(indices) > 0
        assert len(indices) == 0 or len(indices) == self.groups

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook,
            rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device) if self.training else None
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, indices = chunk_indices, **forward_kwargs) for rvq, chunk, chunk_indices in zip_longest(self.rvqs, x, indices))
        out = tuple(zip(*out))

        # if returning cross entropy loss to rvq codebooks

        if return_ce_loss:
            quantized, ce_losses = out
            return torch.cat(quantized, dim = split_dim), sum(ce_losses)

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, commit_losses, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)

        ret = (quantized, all_indices, commit_losses, *maybe_all_codes)
        return ret
