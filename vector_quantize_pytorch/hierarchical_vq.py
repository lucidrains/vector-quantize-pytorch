from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


def exists(v):
    return v is not None


class _Phi2D(nn.Module):
    def __init__(self, dim: int, resi_ratio: float):
        super().__init__()
        self.resi_ratio = float(abs(resi_ratio))
        self.conv = nn.Conv2d(dim, dim, 3, padding = 1)

    def forward(self, x):
        if self.resi_ratio <= 1e-8:
            return x
        return (1. - self.resi_ratio) * x + self.resi_ratio * self.conv(x)


class HierarchicalVQ(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        scales: Sequence[int],
        decay: float = 0.99,
        commitment_weight: float = 1.,
        rotation_trick: bool = False,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.1,
        orthogonal_reg_weight: float = 0.,
        orthogonal_reg_max_codes: int = 128,
        orthogonal_reg_active_codes_only: bool = False,
        quant_resi: float = 0.5,
        share_quant_resi: int = 1,
        accept_image_fmap: bool = False
    ):
        super().__init__()
        assert accept_image_fmap, 'HierarchicalVQ currently expects accept_image_fmap = True'

        scales = [int(scale) for scale in scales]
        assert len(scales) > 0
        assert scales == sorted(scales)
        assert all(scale > 0 for scale in scales)

        self.dim = dim
        self.scales = tuple(scales)
        self.accept_image_fmap = True

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
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
            accept_image_fmap = True
        )

        if share_quant_resi == 1:
            self.phi_shared = _Phi2D(dim, quant_resi)
            self.phi_levels = None
        else:
            num_phi_levels = len(self.scales) if share_quant_resi <= 0 else min(len(self.scales), int(share_quant_resi))
            self.phi_shared = None
            self.phi_levels = nn.ModuleList([_Phi2D(dim, quant_resi) for _ in range(num_phi_levels)])

    def _choose_phi(self, scale_index: int):
        if exists(self.phi_shared):
            return self.phi_shared

        assert exists(self.phi_levels)

        if len(self.phi_levels) == len(self.scales):
            return self.phi_levels[scale_index]

        if len(self.scales) == 1:
            return self.phi_levels[0]

        position = scale_index / float(len(self.scales) - 1)
        phi_index = round(position * (len(self.phi_levels) - 1))
        phi_index = max(0, min(len(self.phi_levels) - 1, phi_index))
        return self.phi_levels[phi_index]

    def _upsample_to_full(self, q, full_hw, scale_index: int):
        if q.shape[-2:] != full_hw:
            q = F.interpolate(q, size = full_hw, mode = 'bilinear', align_corners = False)

        phi = self._choose_phi(scale_index)
        if exists(phi):
            q = phi(q)

        return q

    def forward(
        self,
        x,
        indices = None,
        sample_codebook_temp = None,
        **kwargs
    ):
        assert indices is None, 'reconstruction-from-indices path not implemented in forward'
        del kwargs

        assert x.ndim == 4, 'expected image fmap of shape (batch, channels, height, width)'
        batch, dim, height, width = x.shape
        assert dim == self.dim

        residual = x
        reconstruction = torch.zeros_like(x)
        all_indices = []
        all_commit_losses = []

        for scale_index, scale in enumerate(self.scales):
            residual_down = F.adaptive_avg_pool2d(residual, output_size = (scale, scale))

            vq_kwargs = {}
            if exists(sample_codebook_temp):
                vq_kwargs.update(sample_codebook_temp = sample_codebook_temp)

            quantized, scale_indices, commit_loss = self.vq(residual_down, **vq_kwargs)
            quantized = self._upsample_to_full(quantized, (height, width), scale_index)

            reconstruction = reconstruction + quantized
            residual = residual - quantized

            all_indices.append(scale_indices)
            all_commit_losses.append(commit_loss)

        mean_commit_loss = torch.stack(all_commit_losses).mean()
        return reconstruction, tuple(all_indices), mean_commit_loss

    def get_output_from_indices(self, indices):
        assert isinstance(indices, (tuple, list))
        assert len(indices) == len(self.scales)

        first = indices[0]
        assert first.ndim == 3
        batch, height, width = first.shape
        full_hw = (self.scales[-1], self.scales[-1])
        if (height, width) != full_hw:
            full_hw = (self.scales[-1], self.scales[-1])

        reconstructed = None

        for scale_index, scale_indices in enumerate(indices):
            q = self.vq.get_output_from_indices(scale_indices)
            q = self._upsample_to_full(q, full_hw, scale_index)
            reconstructed = q if reconstructed is None else reconstructed + q

        return reconstructed
