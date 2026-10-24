from __future__ import annotations

import random
from typing import Sequence

import einx
import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from torch_einops_utils.shape import assert_shape, shape
from torch_einops_utils.torch_einops_utils import exists, lens_to_mask, masked_mean

from vector_quantize_pytorch.residual_vq import get_maybe_sync_seed
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


def to_level_shape(scale, spatial_dims):
    if isinstance(scale, int):
        return (scale,) * spatial_dims

    assert len(scale) == spatial_dims
    return tuple(scale)


class _Phi(nn.Module):
    # residual conv after upsampling

    def __init__(self, dim, spatial_dims, resi_ratio):
        super().__init__()
        conv_cls = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[spatial_dims - 1]
        self.conv = conv_cls(dim, dim, 3, padding = 1)
        self.resi_ratio = float(abs(resi_ratio))

    def forward(self, x):
        if self.resi_ratio <= 1e-8:
            return x
        return x.lerp(self.conv(x), self.resi_ratio)


class HierarchicalVQ(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int | tuple[int, ...],
        scales: Sequence[int | Sequence[int]],
        shared_codebook = True,
        channel_first = True,
        sync_codebook = None,
        decay = 0.99,
        commitment_weight = 1.,
        rotation_trick = False,
        kmeans_init = True,
        kmeans_iters = 10,
        threshold_ema_dead_code = 2,
        stochastic_sample_codes = False,
        sample_codebook_temp = 0.1,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_max_codes = 128,
        orthogonal_reg_active_codes_only = False,
        quant_resi = 0.5,
        share_quant_resi = 1,
        nested_dropout = 0.,
        nested_dropout_min_levels = 1,
        accept_image_fmap = False,
        accept_3d_fmap = False
    ):
        super().__init__()
        assert not (accept_image_fmap and accept_3d_fmap), 'choose at most one of accept_image_fmap / accept_3d_fmap'
        assert 0. <= nested_dropout <= 1.
        assert nested_dropout_min_levels >= 1

        spatial_dims = 2 if accept_image_fmap else 3 if accept_3d_fmap else 1

        # scale -> per-level shape
        level_shapes = tuple(to_level_shape(scale, spatial_dims) for scale in scales)

        assert level_shapes
        assert all(s > 0 for level_shape in level_shapes for s in level_shape)

        for prev, curr in zip(level_shapes, level_shapes[1:]):
            assert all(a <= b for a, b in zip(prev, curr)), 'scales must be non-decreasing'

        codebook_sizes = None

        if isinstance(codebook_size, int):
            codebook_sizes = (codebook_size,) * len(level_shapes)
        else:
            codebook_sizes = tuple(codebook_size)

        assert len(codebook_sizes) == len(level_shapes)

        if shared_codebook:
            assert len(set(codebook_sizes)) == 1, 'shared codebook requires uniform codebook sizes'

        self.dim = dim
        self.spatial_dims = spatial_dims
        self.channel_first = channel_first
        self.scales = tuple(scales)
        self.level_shapes = level_shapes
        self.num_levels = len(level_shapes)
        self.codebook_sizes = codebook_sizes
        self.nested_dropout = nested_dropout
        self.nested_dropout_min_levels = nested_dropout_min_levels

        # rearrange layers

        spatial_axes = ('l', 'h w', 'd h w')[spatial_dims - 1]
        axis_names = spatial_axes.split()

        self.to_channel_first = Rearrange('b ... c -> b c ...') if not channel_first else nn.Identity()
        self.to_channel_last = Rearrange('b c ... -> b ... c') if not channel_first else nn.Identity()
        self.mask_to_channel = Rearrange('b ... -> b 1 ...')
        self.mask_from_channel = Rearrange('b 1 ... -> b ...')

        # flatten spatial dims for uniform masks / lens

        self.flatten_spatial = Rearrange('b c ... -> b (...) c')
        self.flatten_indices = Rearrange('b ... -> b (...)')
        self.unflatten_spatial = nn.ModuleList([
            Rearrange(f'b ({spatial_axes}) c -> b c {spatial_axes}', **dict(zip(axis_names, level_shape)))
            for level_shape in level_shapes
        ])
        self.indices_to_spatial = nn.ModuleList([
            Rearrange(f'b ({spatial_axes}) -> b {spatial_axes}', **dict(zip(axis_names, level_shape)))
            for level_shape in level_shapes
        ])

        vq_kwargs = dict(
            dim = dim,
            decay = decay,
            commitment_weight = commitment_weight,
            rotation_trick = rotation_trick,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            threshold_ema_dead_code = threshold_ema_dead_code,
            stochastic_sample_codes = stochastic_sample_codes,
            sample_codebook_temp = sample_codebook_temp,
            orthogonal_reg_weight = orthogonal_reg_weight,
            orthogonal_reg_max_codes = orthogonal_reg_max_codes,
            orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only,
            sync_codebook = sync_codebook,
            channel_last = True
        )

        self.vq = None
        self.vqs = None

        if shared_codebook:
            self.vq = VectorQuantize(codebook_size = codebook_sizes[0], **vq_kwargs)
        else:
            self.vqs = nn.ModuleList([
                VectorQuantize(codebook_size = size, **vq_kwargs)
                for size in codebook_sizes
            ])

        # shared or interpolated residual convs

        self.phi_shared = None
        self.phi_levels = None

        if share_quant_resi == 1:
            self.phi_shared = _Phi(dim, spatial_dims, quant_resi)
        else:
            num_phi_levels = self.num_levels if share_quant_resi <= 0 else min(self.num_levels, int(share_quant_resi))
            self.phi_levels = nn.ModuleList([_Phi(dim, spatial_dims, quant_resi) for _ in range(num_phi_levels)])

    def get_vq(self, level_index):
        return self.vq if exists(self.vq) else self.vqs[level_index]

    def pool_residual(self, t, level_shape, mask = None):
        pool = getattr(F, f'adaptive_avg_pool{self.spatial_dims}d')

        if not exists(mask):
            return pool(t, output_size = level_shape), None

        mask = self.mask_to_channel(mask).float()
        pooled_mask = pool(mask, output_size = level_shape)

        # coarsest level -> global masked mean

        if all(s == 1 for s in level_shape):
            pooled = masked_mean(t, mask = mask, dim = slice(2, None), keepdim = True)
        else:
            pooled = pool(t * mask, output_size = level_shape) / pooled_mask.clamp(min = 1e-6)

        return pooled, self.mask_from_channel(pooled_mask) > 0

    def choose_phi(self, level_index):
        if exists(self.phi_shared):
            return self.phi_shared

        num_phi = len(self.phi_levels)
        if num_phi == 1:
            return self.phi_levels[0]

        pos = level_index / max(self.num_levels - 1, 1)
        return self.phi_levels[round(pos * (num_phi - 1))]

    def upsample_to_full(self, q, full_shape, level_index):
        if tuple(shape(q, 'b c ... -> ...')) != full_shape:
            mode = ('linear', 'bilinear', 'trilinear')[self.spatial_dims - 1]
            q = F.interpolate(q, size = full_shape, mode = mode, align_corners = False)

        phi = self.choose_phi(level_index)
        return phi(q) if exists(phi) else q

    def sample_nested_cutoff(self, device):
        if not self.training or self.nested_dropout <= 0. or self.num_levels <= 1:
            return None

        min_levels = self.nested_dropout_min_levels
        if min_levels >= self.num_levels:
            return None

        seed = get_maybe_sync_seed(device)
        rand = random.Random(seed)

        if rand.random() >= self.nested_dropout:
            return None

        # keep levels 0..cutoff
        return rand.randrange(min_levels - 1, self.num_levels - 1)

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        lens = None,
        sample_codebook_temp = None,
        return_cutoff = False
    ):
        assert not exists(indices), 'reconstruction-from-indices path not implemented in forward'
        assert not (exists(mask) and exists(lens)), 'pass either mask or lens'

        if self.channel_first:
            assert_shape(x, 'b c ...', c = self.dim)
            full_shape = tuple(shape(x, 'b c ... -> ...'))
        else:
            assert_shape(x, 'b ... c', c = self.dim)
            full_shape = tuple(shape(x, 'b ... c -> ...'))

        assert len(full_shape) == self.spatial_dims, f'expected {self.spatial_dims} spatial dims'

        if exists(lens):
            assert self.spatial_dims == 1, 'lens is only supported for 1d sequences'
            mask = lens_to_mask(lens, full_shape[0]).to(x.device)

        if exists(mask):
            assert tuple(shape(mask, 'b ...').ellipsis) == full_shape, 'mask must match spatial shape'
            mask = mask.to(x.device)

        x = self.to_channel_first(x)

        vq_kwargs = dict(sample_codebook_temp = sample_codebook_temp) if exists(sample_codebook_temp) else dict()

        residual = x
        reconstruction = torch.zeros_like(x)
        cumulative = []
        all_indices = []
        commit_losses = []

        for level_index, level_shape in enumerate(self.level_shapes):
            residual_down, level_mask = self.pool_residual(residual, level_shape, mask)
            level_mask = self.flatten_indices(level_mask) if exists(level_mask) else None

            quantized, level_indices, commit_loss = self.get_vq(level_index)(
                self.flatten_spatial(residual_down),
                mask = level_mask,
                **vq_kwargs
            )

            quantized = self.unflatten_spatial[level_index](quantized)
            quantized = self.upsample_to_full(quantized, full_shape, level_index)
            level_indices = self.indices_to_spatial[level_index](level_indices)

            reconstruction = reconstruction + quantized
            residual = residual - quantized

            cumulative.append(reconstruction)
            all_indices.append(level_indices)
            commit_losses.append(commit_loss)

        cutoff = self.sample_nested_cutoff(x.device)

        if exists(cutoff):
            reconstruction = cumulative[cutoff]
            kept_levels = torch.arange(self.num_levels, device = x.device) <= cutoff
        else:
            kept_levels = None

        commit_loss = masked_mean(torch.stack(commit_losses), mask = kept_levels)

        if exists(mask):
            reconstruction = einx.where('b ..., b c ..., -> b c ...', mask, reconstruction, 0.)

        reconstruction = self.to_channel_last(reconstruction)

        if return_cutoff:
            return reconstruction, tuple(all_indices), commit_loss, cutoff

        return reconstruction, tuple(all_indices), commit_loss

    def get_output_from_indices(self, indices, full_shape = None):
        assert isinstance(indices, (tuple, list))
        assert len(indices) == self.num_levels
        assert len(shape(indices[0], 'b ...').ellipsis) == self.spatial_dims

        full_shape = self.level_shapes[-1] if not exists(full_shape) else to_level_shape(full_shape, self.spatial_dims)
        assert len(full_shape) == self.spatial_dims

        reconstructed = None

        for level_index, level_indices in enumerate(indices):
            flat_indices = self.flatten_indices(level_indices)

            q = self.get_vq(level_index).get_output_from_indices(flat_indices)
            q = einx.where('b n, b n c, -> b n c', flat_indices != -1, q, 0.)
            q = self.unflatten_spatial[level_index](q)
            q = self.upsample_to_full(q, full_shape, level_index)

            reconstructed = q if reconstructed is None else reconstructed + q

        return self.to_channel_last(reconstructed)
