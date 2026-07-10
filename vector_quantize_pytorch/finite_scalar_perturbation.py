"""
Finite Scalar Perturbation (FSP)
https://arxiv.org/abs/2602.17133

Each scalar is mapped to [0, 1] via a CDF activation, then quantized into
discrete bins. During training, stochastic perturbation within bins provides
smoother gradient signals and regularization.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from einops import rearrange
from torch import int32, nn
from torch.nn import Module


def default(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


# CDF activation functions: (-inf, +inf) -> [0, 1]
# inv activation functions: (0, 1) -> (-∞, +∞)

def tanh_act(z: torch.Tensor) -> torch.Tensor:
    return (torch.tanh(z) + 1.0) / 2.0


def tanh_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.arctanh(p * 2.0 - 1.0)


def sigmoid_act(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def sigmoid_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p)


_SQRT2 = math.sqrt(2.0)


def normal_act(z: torch.Tensor) -> torch.Tensor:
    return (1.0 + torch.erf(z / _SQRT2)) / 2.0


def normal_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(2.0 * p - 1.0) * _SQRT2


def laplace_act(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.sign(z) * (1.0 - torch.exp(-torch.abs(z))))


def laplace_inv_act(p: torch.Tensor) -> torch.Tensor:
    return -torch.sign(p - 0.5) * torch.log(1.0 - 2.0 * torch.abs(p - 0.5))


def cauchy_act(z: torch.Tensor) -> torch.Tensor:
    return torch.arctan(z) / torch.pi + 0.5


def cauchy_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.tan((p - 0.5) * torch.pi)


_CDF_REGISTRY = {
    "tanh": (tanh_act, tanh_inv_act),
    "sigmoid": (sigmoid_act, sigmoid_inv_act),
    "normal": (normal_act, normal_inv_act),
    "laplace": (laplace_act, laplace_inv_act),
    "cauchy": (cauchy_act, cauchy_inv_act),
}


def build_cdf_act(act_name: str) -> tuple[Callable, Callable]:
    assert act_name in _CDF_REGISTRY, (
        f"CDF activation {act_name} not available: {list(_CDF_REGISTRY.keys())}"
    )
    return _CDF_REGISTRY[act_name]


# batch statistics


def batch_stats(batch: torch.Tensor, eps: float = 1e-8):
    variance, mean = torch.var_mean(batch, dim=0, unbiased=True)
    std = variance.sqrt().clamp_min(eps)
    z_scores = (batch - mean) / std
    skewness = torch.mean(z_scores.pow(3), dim=0)
    kurtosis = torch.mean(z_scores.pow(4), dim=0) - 3.0
    return mean, variance, skewness, kurtosis


# VectorNorm — statistical regularization


class VectorNorm(nn.Module):
    def __init__(
        self,
        l1_target: float = 0.0,
        l1_weight: float = 0.1,
        l2_target: float = 1.0,
        l2_weight: float = 0.07,
        l3_target: float = 0.0,
        l3_weight: float = 0.06,
        l4_target: float = 0.0,
        l4_weight: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.l1_target, self.l1_weight = l1_target, l1_weight
        self.l2_target, self.l2_weight = l2_target, l2_weight
        self.l3_target, self.l3_weight = l3_target, l3_weight
        self.l4_target, self.l4_weight = l4_target, l4_weight
        self.eps = eps

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mean, variance, skewness, kurtosis = batch_stats(z, self.eps)
        norm_loss = (
            ((mean - self.l1_target) ** 2).mean() * self.l1_weight
            + ((variance - self.l2_target) ** 2).mean() * self.l2_weight
            + ((skewness - self.l3_target) ** 2).mean() * self.l3_weight
            + ((kurtosis - self.l4_target) ** 2).mean() * self.l4_weight
        )

        return norm_loss, {
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    @classmethod
    def build(cls, name):
        presets = {
            "none": dict(
                l1_weight=0.0,
                l2_weight=0.0,
                l3_weight=0.0,
                l4_weight=0.0,
            ),
            "var": dict(
                l1_target=0.0,
                l1_weight=0.1,
                l2_target=1.0,
                l2_weight=0.07,
                l3_weight=0.0,
                l4_weight=0.0,
            ),
            "kurt": dict(
                l1_target=0.0,
                l1_weight=0.1,
                l2_target=1.0,
                l2_weight=0.07,
                l3_target=0.0,
                l3_weight=0.06,
                l4_target=0.0,
                l4_weight=0.05,
            ),
            "var_tanh": dict(
                l1_target=0.0,
                l1_weight=0.1,
                l2_target=0.8225,
                l2_weight=0.07,
                l3_weight=0.0,
                l4_weight=0.0,
            ),
            "var_sigmoid": dict(
                l1_target=0.0,
                l1_weight=0.1,
                l2_target=3.29,
                l2_weight=0.07,
                l3_weight=0.0,
                l4_weight=0.0,
            ),
            "var_laplace": dict(
                l1_target=0.0,
                l1_weight=0.1,
                l2_target=2.0,
                l2_weight=0.07,
                l3_weight=0.0,
                l4_weight=0.0,
            ),
        }

        assert name in presets, (
            f"unknown vector_norm preset: {name}, available: {list(presets.keys())}"
        )
        return cls(**presets[name])


# FSP


class FSP(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        channel_first=False,
        projection_has_bias=True,
        act_name="tanh",
        quantize_rate=0.0,
        need_inv_act=False,
        vector_norm="var_tanh",
    ):
        super().__init__()

        # Input validation
        assert 0.0 <= quantize_rate <= 1.0, (
            f"quantize_rate must be in [0.0, 1.0], got {quantize_rate}"
        )

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.dim = default(dim, self.codebook_dim)
        self.channel_first = channel_first

        # levels and mixed-radix basis

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + list(levels[:-1])), dim=0, dtype=int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_size = _levels.prod().item()

        # projections
        self.has_projections = self.dim != self.codebook_dim
        if self.has_projections:
            self.project_in = nn.Linear(
                self.dim, self.codebook_dim, bias=projection_has_bias
            )
            self.project_out = nn.Linear(
                self.codebook_dim, self.dim, bias=projection_has_bias
            )
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        # CDF activation
        self.act_name = act_name
        self.act_func, self.inv_act_func = build_cdf_act(act_name)
        self.need_inv_act = need_inv_act

        self.quantize_rate = quantize_rate

        self.vector_norm = VectorNorm.build(vector_norm)

    def __repr__(self):
        return (
            f"FSP(\n"
            f"  levels={self._levels.tolist()},\n"
            f"  codebook_size={self.codebook_size},\n"
            f"  codebook_dim={self.codebook_dim},\n"
            f"  dim={self.dim},\n"
            f"  act_name='{self.act_name}',\n"
            f"  need_inv_act={self.need_inv_act},\n"
            f"  quantize_rate={self.quantize_rate}\n"
            f")"
        )

    def quantize_act_value(self, act_z, eps):
        # quantize [0,1] values to bin midpoints with straight-through gradients
        level_indices = (act_z.clamp_max(1.0 - eps) * self._levels).floor()
        q_act_z = (level_indices + 0.5) / self._levels
        q_act_z = act_z + (q_act_z - act_z).detach()
        return q_act_z, level_indices.detach()

    def level_indices_to_indices(self, level_indices):
        return (level_indices * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, "... -> ... 1")
        return (indices // self._basis) % self._levels

    def indices_to_act_value(self, indices: torch.Tensor):
        level_indices = self.indices_to_level_indices(indices).float()
        return (level_indices + 0.5) / self._levels

    def indices_to_codes(self, indices: torch.Tensor, eps: float = 1e-6):
        q_act_z = self.indices_to_act_value(indices)
        if self.need_inv_act:
            q_z = self.inv_act_func(q_act_z.clamp(eps, 1.0 - eps))
        else:
            # q_z = q_act_z * 2 - 1
            q_z = (q_act_z - 0.5) / 0.28867513459481287

        codes = self.project_out(q_z)

        if self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(
        self, z, eps: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        eps = eps or torch.finfo(z.dtype).eps

        if self.channel_first:
            z = rearrange(z, "b d ... -> b ... d")
        z_shape = z.shape
        assert z_shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but found dimension of {z_shape[-1]}"
        )
        z = z.reshape(-1, self.dim)
        z = self.project_in(z)

        norm_loss, norm_info = self.vector_norm(z)

        # CDF activation: map to [0, 1]
        act_z = self.act_func(z)
        q_act_z, level_indices = self.quantize_act_value(act_z, eps=eps)
        other_info = {}

        quantize_rate = self.quantize_rate if self.training else 1.0

        if quantize_rate < 1.0:
            p_max_norm = 1.0 / (self._levels * 2)
            perturbations = p_max_norm * (torch.rand_like(act_z) * 2.0 - 1.0)
            proposal = act_z + perturbations
            accept_mask = (proposal > 0.0) & (proposal < 1.0)
            other_info["p_accept_prob"] = accept_mask.float().mean()
            p_act_z = torch.where(accept_mask, proposal, act_z)

            p_mask = torch.rand_like(q_act_z) > quantize_rate
            q_act_z = torch.where(p_mask, p_act_z, q_act_z)

        if self.need_inv_act:
            q_z = self.inv_act_func(q_act_z.clamp(eps, 1.0 - eps))
            q_z = z + (q_z - z).detach()
        else:
            # q_z = q_act_z * 2 - 1
            q_z = (q_act_z - 0.5) / 0.28867513459481287  # to make q_z.var() -> 1.0

        indices = self.level_indices_to_indices(level_indices)
        q_z = self.project_out(q_z)

        level_indices = level_indices.reshape(z_shape[:-1] + (-1,))
        indices = indices.reshape(z_shape[:-1])
        q_z = q_z.reshape(z_shape)
        if self.channel_first:
            q_z = rearrange(q_z, "b ... d -> b d ...")

        return q_z, indices, norm_loss, {
            "level_indices": level_indices,
            'norm_info': norm_info,
            **other_info
        }
