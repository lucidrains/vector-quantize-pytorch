import torch
import torch.nn as nn
from torch.distributions import normal
from typing import Tuple
import warnings
import math
from einops import rearrange, pack, unpack

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

class SFDIVEQ(nn.Module):
    """
    SF-DiVeQ: Space-Filling Differentiable Vector Quantization (VQ) module that allows
    end-to-end training of VQ-based models without any auxiliary losses or
    hyperparameter tunings. The module does not need codebook reset, because its
    training strategy inherently pulls all codewords inside the latent space.
    The module uses a custom initialization such that it skips quantization for the
    first `skip_iters` iterations, and then it initializes the codebook by the recent
    latent vectors and starts quantizing the latents.

    Args:
        - codebook_size (int): No. of codewords.
        - dim (int): Dimensionality of embeddings.
        - noise_var (float): Variance of the directional noise in SF-DiVeQ.
            Recommended noise_var < 1e-2.
        - skip_iters (int): Number of training iterations to skip quantization for
            SF-DiVeQ custom initialization. Recommended skip_iters > 1000.
        - avg_iters (int): Number of recent training iterations to extract continuous
            latents for SF-DiVeQ custom codebook initialization, before starting
            quantization. Recommended  50 < avg_iters < 100.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
            If False, the codebook is initialized from a normal distribution.
        - allow_warning (bool): Whether to print the warnings.
        - latents_on_cpu (bool): Whether to collect latents for initialization on cpu.
            If running out of CUDA memory, set it to True.
        - channel_first (bool): Whether the first dimension of input refers to Channel;
            It means whether z.shape = (B, Channels, H, W) or (B, Channels, N=HxW)

    Returns:
        - z_q (torch.Tensor): Differentiable quantized input/latent.
            Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
        - indices (torch.Tensor): Selected codebook indices. Shape (B, H, W) or (B, N=HxW)
        - perplexity (float): Codebook usage (entropy of selected indices).
    """
    def __init__(
            self,
            codebook_size: int,
            dim: int,
            noise_var: float = 0.001,
            skip_iters: int = 1000,
            avg_iters: int = 50,
            uniform_init: bool = True,
            allow_warning: bool = True,
            latents_on_cpu = True,
            channel_first = False,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.dim = dim
        self.noise_var = noise_var
        self.skip_iters = skip_iters
        self.avg_iters = avg_iters
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.latents_on_cpu = latents_on_cpu
        self.channel_first = channel_first

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if noise_var > 0.01:
                warnings.warn(f"`noise_var` is set to {noise_var}, which is"
                              f" quite large. Values > 0.01 may overshoot"
                              f" nearest-neighbor mapping.", UserWarning)

            if skip_iters < 1000:
                warnings.warn(f"`skip_iters` is set to {skip_iters}, which is"
                              f" quite small. Set `skip_iters` to enough large values"
                              f" that ensure the latent space has passed its initial"
                              f" big mutations.", UserWarning)

            if avg_iters < 50:
                warnings.warn(f"`avg_iters` is set to {avg_iters}, which is"
                              f" quite small. Values < 50 may result in a small pool of"
                              f" latent vectors for custom initialization."
                              f" If running out of memory (OOM), set a smaller value or"
                              f" set `latents_on_cpu` to True.", UserWarning)
            elif avg_iters > 100:
                warnings.warn(f"`avg_iters` is set to {avg_iters}, which is"
                              f" quite large. Values > 100 may result in an out-dated"
                              f" pool of latent vectors for custom initialization."
                              f" If running out of memory (OOM), set a smaller value or"
                              f" set `latents_on_cpu` to True.",UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.codebook_size, self.dim))
                        * (1 / self.codebook_size))
        else:
            codebook = (torch.randn((self.codebook_size, self.dim))
                        * (1 / self.codebook_size))

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # --------- Tensors used for SF-DiVeQ custom codebook initialization ---------
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))
        self.latent_list = []

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            - z (torch.Tensor): input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.

        Returns:
            - z_q (torch.Tensor): Differentiable quantized input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
            - indices (torch.Tensor): Selected codebook indices. Shape (B, H, W) or (B, N=HxW)
            - perplexity (float): Codebook usage (entropy of selected indices).
        """
        if self.channel_first:
            z = rearrange(z, 'b d ... -> b ... d')

        self._check_input(z)

        z, inverse_pack = pack_one(z, 'b * d')
        z_shape = z.shape
        z = z.view(-1, self.dim) # flatten the input

        with torch.no_grad():
            self.iter_counter += 1

        # If iteration is larger than `skip_iters`, apply quantization on the latents
        if self.iter_counter.item() >= self.skip_iters:
            # Create dithered codebook
            dithered_codebook, dither = self._generate_dithered_codebook(z.device)

            # Calculate distances
            distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                         + torch.sum(dithered_codebook.pow(2), dim=1)
                         - 2 * torch.matmul(z, dithered_codebook.t()))

            indices = torch.argmin(distances, dim=1)

            # Perplexity Computation
            perplexity = self._compute_perplexity(indices)

            codebook_first = self.codebook[indices]
            codebook_second = self.codebook[indices + 1]
            interp_lambda = dither[indices]

            direction_first = codebook_first - z
            random_vectors1 = (normal.Normal(0, self.noise_var).sample(z.shape)
                               .to(z.device) + direction_first)
            normalized1 = random_vectors1 / torch.linalg.norm(random_vectors1, dim=1,
                                                          keepdim=True).clamp_min(1e-12)

            direction_second = codebook_second - z
            random_vectors2 = (normal.Normal(0, self.noise_var).sample(z.shape)
                               .to(z.device) + direction_second)
            normalized2 = random_vectors2 / torch.linalg.norm(random_vectors2, dim=1,
                                                          keepdim=True).clamp_min(1e-12)

            error_magnitude1 = torch.linalg.norm(codebook_first - z, dim=1,
                                                                        keepdim=True)
            error_magnitude2 = torch.linalg.norm(codebook_second - z, dim=1,
                                                                        keepdim=True)
            vq_error1 = error_magnitude1 * ((1 - interp_lambda) * normalized1).detach()
            vq_error2 = error_magnitude2 * (interp_lambda * normalized2).detach()

            z_q = z + vq_error1 + vq_error2 # Differentiable quantized input

            z_q = z_q.view(z_shape)
            z_q = inverse_pack(z_q)
            indices = indices.view(z_q.shape[:-1])

            if self.channel_first:
                z_q = rearrange(z_q, 'b ... d-> b d ...')

            return z_q, indices, perplexity

        # If iteration is smaller than `skip_iters`, skip quantization
        else:
            # to capture latents of the last `avg_iters` iterations for
            # SF-DiVeQ custom codebook initialization
            if self.iter_counter.item() >= (self.skip_iters - self.avg_iters):
                if self.latents_on_cpu:
                    self.latent_list.append(z.detach().cpu())
                else:
                    self.latent_list.append(z.detach())
                # to compute and set the initial codebook
                if self.iter_counter.item() == (self.skip_iters - 1):
                    stacked_latents = torch.cat(self.latent_list[:], dim=0)
                    initial_codebook = torch.zeros((self.codebook_size,
                            self.dim), device=self.latent_list[0].device,
                                                   dtype=self.codebook.dtype)
                    hop_size = int(math.floor(stacked_latents.shape[0]
                                              / self.codebook_size))
                    for j in range(self.codebook_size):
                        initial_codebook[j] = torch.mean(stacked_latents[j * hop_size
                                                         :(j + 1) * hop_size], dim=0)
                    if self.latents_on_cpu:
                        self.codebook.data = initial_codebook.to(z.device).clone()
                    else:
                        self.codebook.data = initial_codebook.clone()

            # as placeholder only
            indices = torch.zeros((z.shape[0],), device=z.device, dtype=torch.int64)
            perplexity = torch.tensor(0.0, device=z.device,
                                      dtype=torch.float32).item()

            z = z.view(z_shape)
            z = inverse_pack(z)
            indices = indices.view(z.shape[:-1])

            if self.channel_first:
                z = rearrange(z, 'b ... d-> b d ...')

            return z, indices, perplexity

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input/latent to the nearest point
         on the SF-DiVeQ's curve.
        Args:
            - z (torch.Tensor): input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
            - indices (torch.Tensor): Selected codebook indices. Shape (B, H, W) or (B, N=HxW)
            - perplexity (float): Codebook usage (entropy of selected indices).
        """
        if self.channel_first:
            z = rearrange(z, 'b d ... -> b ... d')

        self._check_input(z)

        z, inverse_pack = pack_one(z, 'b * d')
        z_shape = z.shape
        z = z.view(-1, self.dim) # flatten the input

        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(z, self.codebook.t()))

        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        indices_clamped = indices.clamp(min=1, max=self.codebook_size - 2)

        cm = self.codebook[indices_clamped - 1]
        cc = self.codebook[indices_clamped]
        cp = self.codebook[indices_clamped + 1]

        remainder_index_m = (((cc - cm) * (z - cm)).sum(dim=1) / (cc - cm).square()
                             .sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        remainder_index_p = (((cp - cc) * (z - cc)).sum(dim=1) / (cp - cc).square()
                             .sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        z_q_m = ((1 - remainder_index_m) * cm) + (remainder_index_m * cc)
        z_q_p = ((1 - remainder_index_p) * cc) + (remainder_index_p * cp)
        distance_m = (z - z_q_m).square().sum(dim=1)
        distance_p = (z - z_q_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int64).squeeze() - 1

        c1 = self.codebook[indices_clamped + offset]
        c2 = self.codebook[indices_clamped + offset + 1]

        remainder_index = (((c2 - c1) * (z - c1)).sum(dim=1) / (c2 - c1).square()
                           .sum(dim=1)).clamp(min=0., max=1.)
        z_q_hard = c1 + (remainder_index.reshape(-1, 1) * (c2 - c1))

        z_q_hard = z_q_hard.view(z_shape)
        z_q_hard = inverse_pack(z_q_hard)
        indices = indices.view(z_q_hard.shape[:-1])

        if self.channel_first:
            z_q_hard = rearrange(z_q_hard, 'b ... d-> b d ...')

        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self,) -> None:
        if self.noise_var <= 0.0:
            raise ValueError("`noise_var` must be a positive float value. To have more"
                             " precise nearest-neighbor assignments, it is recommended"
                             " that noise_var < 1e-2.")
        if self.skip_iters < 0:
            raise ValueError("`skip_iters` must be a positive integer value."
                             " It is recommended that skip_iters > 1000.")
        if self.avg_iters < 0:
            raise ValueError("`avg_iters` must be a positive integer value."
                             " It is recommended that 50 < avg_iters < 100.")
        if self.avg_iters > self.skip_iters:
            raise ValueError("`avg_iters` must be smaller than `skip_iters`.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim <= 2:
            raise ValueError("SF-DiVeQ input must have more than two dimensions with"
                             " the shape of (B, H, W, C) or (B, N, C) or (B, C, N) where"
                             " B is the Batch size, C refers to number of Channels,"
                             " N=HxW are the Height and Width in case of image data.")
        if z.size(-1) != self.dim:
            raise ValueError(f"SF-DiVeQ input.size(-1) must match the embedding"
                             f" dimensionality that is {self.dim}.")

    def _generate_dithered_codebook(self, device:str)->Tuple[torch.Tensor,torch.Tensor]:
        dither = torch.rand((self.codebook_size - 1, 1), device=device)
        integer_index = torch.linspace(0, self.codebook_size - 2,
                               self.codebook_size - 1, device=device).to(torch.int64)
        c1 = self.codebook[integer_index]
        c2 = self.codebook[integer_index + 1]
        dithered_codebook = ((1 - dither) * c1) + (dither * c2)
        return dithered_codebook, dither

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.codebook_size,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def extra_repr(self) -> str:
        return (
            f"codebook_size={self.codebook_size}, "
            f"dim={self.dim}, "
            f"noise_var={self.noise_var}, "
            f"skip_iters={self.skip_iters}, "
            f"avg_iters={self.avg_iters}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"latents_on_cpu={self.latents_on_cpu}, "
            f"channel_first={self.channel_first}"
        )