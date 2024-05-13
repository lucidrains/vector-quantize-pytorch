"""
Disentanglement via Latent Quantization
 - https://arxiv.org/abs/2305.18378
Code adapted from Jax version in https://github.com/kylehkhsu/latent_quantization
"""

from __future__ import annotations
from typing import Callable, List

import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import Tensor, int32, nn
from torch.nn import Module
from torch.optim import Optimizer

# helper functions


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class LatentQuantize(Module):
    def __init__(
        self,
        levels: List[int] | int,
        dim: int,
        commitment_loss_weight: float | None = 0.1,
        quantization_loss_weight: float | None = 0.1,
        num_codebooks: int = 1,
        codebook_dim: int = -1,
        keep_num_codebooks_dim: bool | None = None,
        optimize_values: bool | None = True,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used to update the codebook embedding if using learnable_codebook
    ):
        """
        Initializes the LatentQuantization module.

        Args:
            levels (List[int]|init): The number of levels per codebook.
                If an int is provided, it is used for all codebooks.
            dim (int): The dimensionality of the input tensor.
                The input tensor is expected to be of shape [B D ...]
            num_codebooks (int): The number of codebooks to use.
                (default is 1)
            codebook_dim (int): the dimension of the codebook.
                If levels is a list, codebook_dim is the length of the list.
                (default to -1) 
            keep_num_codebooks_dim (Optional[bool]): Whether to keep the number of codebooks dimension in the output tensor. If not provided, it is set to True if num_codebooks > 1, otherwise False.
            optimize_values (Optional[bool]): Whether to optimize the values of the codebook. If not provided, it is set to True.
        """
        super().__init__()

        self.dim = dim
        self.in_place_codebook_optimizer = in_place_codebook_optimizer
        _levels = torch.tensor(levels, dtype=int32)

        # if levels is an int, use it for all codebooks
        if isinstance(levels, int):
            try:
                _levels = _levels.repeat(codebook_dim)
            except RuntimeError as e:
                raise e
        self.register_buffer(
            "commitment_loss_weight",
            torch.tensor(commitment_loss_weight, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "quantization_loss_weight",
            torch.tensor(quantization_loss_weight, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.concat([torch.tensor([1], dtype=int32), _levels[:-1]], dim=0), dim=0
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_dim = codebook_dim if codebook_dim > 0 else len(_levels)

        effective_codebook_dim = self.codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim if keep_num_codebooks_dim else num_codebooks > 1
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        values_per_latent = [
            torch.linspace(-0.5, 0.5, level)
            if level % 2 == 1
            else torch.arange(level) / level - 0.5
            for level in _levels
        ]  # ensure zero is in the middle and start is always -0.5

        # test, and check whether it would be in the parameters of the model or not
        if optimize_values:
            self.values_per_latent = nn.ParameterList(
                [nn.Parameter(values) for values in values_per_latent]
            )
            if in_place_codebook_optimizer is not None:
                self.in_place_codebook_optimizer = in_place_codebook_optimizer(
                    self.values_per_latent
                )
        else:
            self.values_per_latent = values_per_latent  # are there any scenarios where this would have its gradients updated?

    def quantization_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction=reduce)

    def commitment_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction=reduce)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """

        def distance(x, y):
            return torch.abs(x - y)

        index = torch.stack(
            [
                torch.argmin(
                    distance(z[..., i, None], self.values_per_latent[i]), dim=-1
                )
                for i in range(self.codebook_dim)
            ],
            dim=-1,
        )
        quantize = torch.stack(
            [
                self.values_per_latent[i][index[..., i]]
                for i in range(self.codebook_dim)
            ],
            dim=-1,
        )

        quantize = z + (quantize - z).detach()
        # half_width = self._levels // 2 / 2  # Renormalize to [-0.5, 0.5].
        return quantize  # / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        """scale and shift zhat from [-0.5, 0.5] to [0, level_per_dim]"""
        half_width = self._levels // 2
        return (zhat_normalized * 2 * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        """normalize zhat to [-0.5, 0.5]"""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width / 2

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def quantize_and_project(self, z: Tensor, is_img_or_video, ps) -> Tensor:
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)
        out = unpack_one(out, ps, "b * d")
        out = rearrange(out, "b ... d -> b d ...")

        indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        return codes, out, indices

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        original_input = z
        should_inplace_optimize = self.in_place_codebook_optimizer is not None

        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        # project in
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)
        out = unpack_one(out, ps, "b * d")
        out = rearrange(out, "b ... d -> b d ...")

        indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        if should_inplace_optimize and self.training and not self.optimize_values:
            # update codebook
            loss = (
                self.commitment_loss(z, out)
                if self.commitment_loss_weight != 0
                else torch.tensor(0.0)
            )
            loss += (
                self.quantization_loss(z, out)
                if self.quantization_loss_weight != 0
                else torch.tensor(0.0)
            )
            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()
            # quantize again
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)
            codes = rearrange(codes, "b n c d -> b n (c d)")
            out = self.project_out(codes)

            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, "... 1 -> ...")

        # calculate losses
        commitment_loss = (
            self.commitment_loss(original_input, out)
            if self.training and self.commitment_loss_weight != 0
            else torch.tensor(0.0)
        )
        quantization_loss = (
            self.quantization_loss(original_input, out)
            if self.training and self.quantization_loss_weight != 0
            else torch.tensor(0.0)
        )

        loss = (
            self.commitment_loss_weight * commitment_loss
            + self.quantization_loss_weight * quantization_loss
        )

        return out, indices, loss
