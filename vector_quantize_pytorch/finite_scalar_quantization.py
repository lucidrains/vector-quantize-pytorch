"""Finite Scalar Quantization module.

Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1.
"""

from __future__ import annotations

import torch
from einops import pack, rearrange, unpack
from torch import Tensor, int32, nn
from torch.cuda.amp import autocast
from torch.nn import Module


def round_ste(features: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = features.round()
    return features + (zhat - features).detach()

class FSQ(Module):
    """Finite Scalar Quantization module.

    This follows directly the idea in the initial paper.

    Attributes
    ----------
    codebook_dim: int
        the dimension of the codebook, that is the length of the list `levels`.
    num_codebooks: int
        the number of codebooks
    effective_codebook_dim: int
        the number of codebooks times the dimension of one codebook.
    keep_num_codebooks_dim: bool
        whether to keep an additional dimension to the return tensors, corresponding to the number of codebooks.
        If `keep_num_codebooks_dim` is not provided at init, it is True if there are several codebooks.
    dim: int
        the dimension of the feature tensor.
        If not provided, it is the effective codebook dimension.
    channel_first: bool
        indicates if the feature tensor has channel first (B, C, *) or last (B, *, C).
    has_projections: bool
        indicates if projections before and after quantization are needed, if the dimension of the features tensor is not the effective codebook dimension.
    return_indices: bool
        indicates if indices are returned or not. Useful in case of very large codebooks.

    Parameters
    ----------
    project_in: nn.Module
        is a Linear module if `dim` is not `effective_codebook_dim`, nn.Identity() else.
    project_out: nn.Module
        is a Linear module if `dim` is not `effective_codebook_dim`, nn.Identity() else.
    
    Methods
    -------
    bound(self, features: Tensor, eps: float = 1e-3)
    quantize(self, features: Tensor)
    codes_to_indices(self, zhat: Tensor)
    indices_to_level_indices(self, indices: Tensor)
    indices_to_codes(self, indices: Tensor)
    forward(self, features: Tensor)
    """

    def __init__(
        self,
        levels: list[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
    ):
        """Initialize the module.

        Parameters
        ----------
        levels : list[int]
            _description_
        dim : int | None, optional
            _description_, by default None
        num_codebooks : int, optional
            _description_, by default 1
        keep_num_codebooks_dim : bool | None, optional
            _description_, by default None
        allowed_dtypes : tuple[torch.dtype, ...], optional
            _description_, by default (torch.float32, torch.float64)
        channel_first : bool, optional
            _description_, by default False
        projection_has_bias : bool, optional
            _description_, by default True
        return_indices : bool, optional
            _description_, by default True
        """
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim if keep_num_codebooks_dim else num_codebooks > 1
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = dim if dim else len(_levels) * num_codebooks

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer(
                "implicit_codebook", implicit_codebook, persistent=False
            )

        self.allowed_dtypes = allowed_dtypes

    def bound(self, features: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `features`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (features + shift).tanh() * half_l - offset

    def quantize(self, features: Tensor) -> Tensor:
        """Quantize features, returns quantized zhat, same shape as features."""
        quantized = round_ste(self.bound(features))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices: Tensor) -> Tensor:
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, codes: Tensor) -> Tensor:
        """Convert a `code` to an index in the codebook."""
        assert codes.shape[-1] == self.codebook_dim
        codes = self._scale_and_shift(codes)
        return (codes * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices: Tensor) -> Tensor:
        """Convert indices to indices at each level, perhaps needed for a transformer with factorized embeddings."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse of `codes_to_indices`."""
        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    @autocast(enabled=False)
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the forward pass to the features tensor.

        Parameters
        ----------
        features : Tensor
            tensor of shape (B, dim, *) if channel_first, (B, *, dim) if not.

        Returns
        -------
        tuple[Tensor, Tensor]
            the tuple is (out, indices), with:
                * out, the quantized version of features, of the same shape
                * indices, if return_indices is True, of the same shape or with an additional dimension if keep_num_codebooks_dim is set to True.

        """
        orig_dtype = features.dtype
        
        if self.channel_first:
            features = rearrange(features, "b d ... -> b ... d")
        
        features, ps = pack([features], "b * d")

        assert (
            features.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {features.shape[-1]}"

        features = self.project_in(features)

        features = rearrange(features, "b n (c d) -> b n c d", c=self.num_codebooks)

        # make sure allowed dtype before quantizing

        if features.dtype not in self.allowed_dtypes:
            features = features.float()

        codes = self.quantize(features)
        indices = None
        if self.return_indices:
            indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        # cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        out = unpack(out, ps, "b * d")[0]
        if self.channel_first:
            out = rearrange(out, "b ... d -> b d ...")

        if self.return_indices:
            indices = unpack(indices, ps, "b * c")[0]

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices
