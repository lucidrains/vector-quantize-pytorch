import torch
import torch.nn as nn
from torch.distributions import normal
import warnings
from typing import Tuple
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

class ResidualSFDIVEQ(nn.Module):
    """
    Residual_SF-DiVeQ: Differentiable Space-Filling Residual Vector Quantization (RVQ)
    module that allows end-to-end training of VQ-based models without any auxiliary
    losses or hyperparameter tunings. The module does not need codebook replacement,
    because its training strategy inherently pulls all codewords inside the latent
    space. The module uses a custom initialization such that it applies Residual_DiVeQ
    for the first `skip_iters` iterations, and then it initializes the codebooks by the
    recent latent vectors and starts Residual_SF-DiVeQ.

    Args:
        - codebook_size (int): No. of codewords.
        - dim (int): Dimensionality of embeddings.
        - num_quantizers (int): No. of codebooks involved in quantization.
        - noise_var (float): Variance of the directional noise in SF-DiVeQ.
            Recommended noise_var < 1e-2.
        - skip_iters (int): Number of training iterations to skip quantization for
            SF-DiVeQ custom initialization. Recommended skip_iters > 1000.
        - avg_iters (int): Number of last training iterations to extract continuous
            latents for SF-DiVeQ custom codebook initialization, before starting
            quantization. Recommended  50 < avg_iters < 100.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement before starting Residual_SF-DiVeQ).
            Recommended 50 < replacement_iters < 100. It is recommended to set
            `replacement_iters` at least 10-20 times smaller than `skip_iters` to
            prevent unused codewords to be involved in custom initialization.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iterations.
            Recommended 0.01 < discard_threshold < 0.05. discard_threshold must be in
            the range of [0,1] such that discard_threshold=0.01 means to discard the
            codebook entries which are used less than 1 percent.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            for codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
            If False, the codebook is initialized from a normal distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.
        - latents_on_cpu (bool): Whether to collect latents for initialization on cpu.
            If running out of CUDA memory, set it to True.
        - channel_first (bool): Whether the first dimension of input refers to Channel;
            It means whether z.shape = (B, Channels, H, W) or (B, Channels, N=HxW)

    Returns:
        - z_q (torch.Tensor): Differentiable quantized input/latent.
            Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
        - indices_all (torch.Tensor): Tensor of selected codebook indices for all quantizers
            with the shape of (num_quantizers, B, H, W) or (num_quantizers, B, N=HxW)
        - perplexity_all (torch.Tensor): Tensor of codebook usage (entropy of selected indices)
            for all quantizers with the shape of (num_quantizers, )
    """
    def __init__(
            self,
            codebook_size: int,
            dim: int,
            num_quantizers: int,
            noise_var: float = 0.001,
            skip_iters: int = 1000,
            avg_iters: int = 50,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose = True,
            latents_on_cpu = True,
            channel_first=False,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.noise_var = noise_var
        self.skip_iters = skip_iters
        self.avg_iters = avg_iters
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose
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
                              f" set `latents_on_cpu` to True.", UserWarning)

            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)

            elif replacement_iters > 100:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 100 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = torch.rand((self.num_quantizers, self.codebook_size,
                                   self.dim)) * (1 / self.codebook_size)
        else:
            codebook = torch.randn((self.num_quantizers, self.codebook_size,
                                    self.dim)) * (1 / self.codebook_size)

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # --------- Tensors used for SF-DiVeQ custom codebook initialization ---------
        self.register_buffer("codebook_usage", torch.zeros(self.num_quantizers,
                                               self.codebook_size, dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))
        self.latent_list = [[] for _ in range(self.num_quantizers)]

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - z (torch.Tensor): input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.

        Returns:
            - z_q (torch.Tensor): Differentiable quantized input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
            - indices_all (torch.Tensor): Tensor of selected codebook indices for all quantizers
                with the shape of (num_quantizers, B, H, W) or (num_quantizers, B, N=HxW)
            - perplexity_all (torch.Tensor): Tensor of codebook usage (entropy of selected indices)
                for all quantizers with the shape of (num_quantizers, )
        """

        with torch.no_grad():
            self.iter_counter += 1

        # If iteration is larger than `skip_iters`, quantize by Residual_SF-DiVeQ
        if self.iter_counter.item() >= self.skip_iters:

            if self.channel_first:
                z = rearrange(z, 'b d ... -> b ... d')
            z_input_shape = z.shape

            self._check_input(z)

            z, inverse_pack = pack_one(z, 'b * d')
            z_shape = z.shape
            z = z.view(-1, self.dim)  # flatten the input

            quantized_input_list = []
            remainder_list = []
            indices_list = []
            perplexity_list = []
            remainder_list.append(z)

            for i in range(self.num_quantizers):
                quantized_input, remainder, indices = self._hard_sfvq(remainder_list[i],
                                                                      self.codebook[i])

                quantized_input_list.append(quantized_input)
                remainder_list.append(remainder)
                perplexity = self._compute_perplexity(indices) # Perplexity Computation
                perplexity_list.append(perplexity)
                indices_list.append(indices.view(z_input_shape[:-1]))

            z_hard_quantized = sum(quantized_input_list[:])

            # DiVeQ technique
            direction = z_hard_quantized - z
            random_vectors = (normal.Normal(0, self.noise_var).sample(z.shape)
                              .to(z.device) + direction)
            normalized = random_vectors / torch.linalg.norm(random_vectors, dim=1,
                                                        keepdim=True).clamp_min(1e-12)
            error_magnitude = torch.linalg.norm(z_hard_quantized - z, dim=1,
                                                                        keepdim=True)

            vq_error = error_magnitude * normalized.detach()
            z_q = z + vq_error  # Differentiable quantized input

            z_q = z_q.view(z_shape)
            z_q = inverse_pack(z_q)

            if self.channel_first:
                z_q = rearrange(z_q, 'b ... d-> b d ...')

            # convert to tensors
            indices_all = torch.stack(indices_list, dim=0)
            perplexity_all = torch.tensor(perplexity_list, dtype=torch.float32)

            return z_q, indices_all, perplexity_all

        # If iteration is smaller than `skip_iters`, quantize by Residual_DiVeQ
        else:
            z_q, indices_all, perplexity_all, remainder_list = self._forward_diveq(z)

            with (torch.no_grad()):
                # to capture latents of the last `avg_iters` iterations for
                # Residual_SF-DiVeQ custom codebook initialization
                if self.iter_counter.item() >= (self.skip_iters - self.avg_iters):
                    for i in range(self.num_quantizers):
                        if self.latents_on_cpu:
                            self.latent_list[i].append(remainder_list[i].detach().cpu())
                        else:
                            self.latent_list[i].append(remainder_list[i].detach())
                    # to compute and set the initial codebook
                    if self.iter_counter.item() == (self.skip_iters - 1):
                        for i in range(self.num_quantizers):
                            stacked_latents = torch.cat(self.latent_list[i][:], dim=0)
                            initial_codebook = torch.zeros((self.codebook_size,
                                    self.dim), device=self.latent_list[i][0]
                                                   .device, dtype=self.codebook.dtype)
                            hop_size = int(math.floor(stacked_latents.shape[0] /
                                                                self.codebook_size))
                            for j in range(self.codebook_size):
                                initial_codebook[j] = torch.mean(stacked_latents[j *
                                                 hop_size:(j + 1) * hop_size], dim=0)
                            if self.latents_on_cpu:
                                self.codebook[i].data = initial_codebook.to(z.device
                                                                            ).clone()
                            else:
                                self.codebook[i].data = initial_codebook.clone()

            return z_q, indices_all, perplexity_all

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic hard quantization by mapping the input/latent to the nearest point
        on the Residual_SF-DiVeQ curve.
        Args:
            - z (torch.Tensor): input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.

        Returns:
            - z_q (torch.Tensor): Hard quantized input/latent.
                Shape (B, H, W, C) or, (B, C, H, W) if `channel_first` = True.
            - indices_all (torch.Tensor): Tensor of selected codebook indices for all quantizers
                with the shape of (num_quantizers, B, H, W) or (num_quantizers, B, N=HxW)
            - perplexity_all (torch.Tensor): Tensor of codebook usage (entropy of selected indices)
                for all quantizers with the shape of (num_quantizers, )
        """
        if self.channel_first:
            z = rearrange(z, 'b d ... -> b ... d')
        z_input_shape = z.shape

        self._check_input(z)

        z, inverse_pack = pack_one(z, 'b * d')
        z_shape = z.shape
        z = z.view(-1, self.dim)  # flatten the input

        quantized_input_list = []
        remainder_list = []
        indices_list = []
        perplexity_list = []
        remainder_list.append(z)

        for i in range(self.num_quantizers):
            quantized_input, remainder, indices, perplexity = self._dithered_inference(
                                                    remainder_list[i], self.codebook[i])

            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            perplexity_list.append(perplexity)
            indices_list.append(indices.view(z_input_shape[:-1]))

        z_q_hard = sum(quantized_input_list[:])

        z_q_hard = z_q_hard.view(z_shape)
        z_q_hard = inverse_pack(z_q_hard)

        if self.channel_first:
            z_q_hard = rearrange(z_q_hard, 'b ... d-> b d ...')

        # convert to tensors
        indices_all = torch.stack(indices_list, dim=0)
        perplexity_all = torch.tensor(perplexity_list, dtype=torch.float32)

        return z_q_hard, indices_all, perplexity_all

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
        if self.replacement_iters > self.skip_iters:
            raise ValueError("`replacement_iters` must be smaller than `skip_iters`."
                             " It is recommended to set `replacement_iters` at least"
                             " 10-20 times smaller than `skip_iters` to prevent unused"
                             " codewords to be involved in custom initialization.")
        if (self.replacement_iters <= 0) or (type(self.replacement_iters) is not int):
            raise ValueError("`replacement_iters` must be a positive integer value."
                             " It is recommended that 50 < replacement_iters < 300.")
        if (self.discard_threshold < 0.0) or (self.discard_threshold > 1.0):
            raise ValueError("`discard_threshold` must be in the range of [0,1]. It is"
                             " recommended that 0.01 < discard_threshold < 0.05,"
                             " such that discard_threshold=0.01 means to discard the"
                             " codebook entries which are used less than 1 percent.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim <= 2:
            raise ValueError("SF-DiVeQ input must have more than two dimensions with"
                             " the shape of (B, H, W, C) or (B, N, C) or (B, C, N) where"
                             " B is the Batch size, C refers to number of Channels,"
                             " N=HxW are the Height and Width in case of image data.")
        if z.size(-1) != self.dim:
            raise ValueError(f"SF-DiVeQ input.size(-1) must match the embedding"
                             f" dimensionality that is {self.dim}.")

    def _generate_dithered_codebook(self, x: torch.Tensor, codebook: torch.Tensor) \
                                                -> Tuple[torch.Tensor, torch.Tensor]:
        dither = torch.rand((self.codebook_size - 1, 1), device=x.device)
        integer_index = torch.linspace(0, self.codebook_size - 2,
                               self.codebook_size - 1, device=x.device).to(torch.int64)
        c1 = codebook[integer_index]
        c2 = codebook[integer_index + 1]
        dithered_codebook = ((1 - dither) * c1) + (dither * c2)
        return dithered_codebook, dither

    def _hard_sfvq(self, x: torch.Tensor, codebook: torch.Tensor) \
                                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dithered_codebook, dither = self._generate_dithered_codebook(x, codebook)
        # Calculate distances
        distances = (torch.sum(x.pow(2), dim=1, keepdim=True)
                     + torch.sum(dithered_codebook.pow(2), dim=1)
                     - 2 * torch.matmul(x, dithered_codebook.t()))

        indices = torch.argmin(distances, dim=1)
        x_quantized = dithered_codebook[indices]
        remainder = x - x_quantized
        return x_quantized, remainder, indices

    @staticmethod
    def _hard_vq(x: torch.Tensor, codebook: torch.Tensor) \
                                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (torch.sum(x.pow(2), dim=1, keepdim=True)
                     + torch.sum(codebook.pow(2), dim=1)
                     - 2 * torch.matmul(x, codebook.t()))
        indices = torch.argmin(distances, dim=1)
        x_quantized = codebook[indices]
        remainder = x - x_quantized
        return x_quantized, remainder, indices

    def _forward_diveq(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:

        if self.channel_first:
            z = rearrange(z, 'b d ... -> b ... d')
        z_input_shape = z.shape

        self._check_input(z)

        z, inverse_pack = pack_one(z, 'b * d')
        z_shape = z.shape
        z = z.view(-1, self.dim)  # flatten the input

        quantized_input_list = []
        remainder_list = []
        indices_list = []
        perplexity_list = []
        remainder_list.append(z)

        for i in range(self.num_quantizers):
            quantized_input, remainder, indices = self._hard_vq(remainder_list[i],
                                                                self.codebook[i])

            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            perplexity = self._compute_perplexity(indices) # Perplexity Computation
            perplexity_list.append(perplexity)

            # Track used indices for codebook replacement
            with torch.no_grad():
                self.codebook_usage[i, indices] += 1

            indices_list.append(indices.view(z_input_shape[:-1]))

        z_hard_quantized = sum(quantized_input_list[:])

        # DiVeQ technique
        direction = z_hard_quantized - z
        random_vectors = (normal.Normal(0, self.noise_var).sample(z.shape)
                          .to(z.device) + direction)
        normalized = random_vectors / torch.linalg.norm(random_vectors, dim=1,
                                                        keepdim=True).clamp_min(1e-12)
        error_magnitude = torch.linalg.norm(z_hard_quantized - z, dim=1, keepdim=True)

        vq_error = error_magnitude * normalized.detach()
        z_q = z + vq_error  # Differentiable quantized input

        z_q = z_q.view(z_shape)
        z_q = inverse_pack(z_q)

        if self.channel_first:
            z_q = rearrange(z_q, 'b ... d-> b d ...')

        # Track used indices for codebook replacement
        with torch.no_grad():
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        # convert to tensors
        indices_all = torch.stack(indices_list, dim=0)
        perplexity_all = torch.tensor(perplexity_list, dtype=torch.float32)

        return z_q, indices_all, perplexity_all, remainder_list

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.codebook_size,
                                                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _dithered_inference(self, x: torch.Tensor, codebook: torch.Tensor) \
                            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        distances = (torch.sum(x.pow(2), dim=1, keepdim=True)
                     + torch.sum(codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(x, codebook.t()))

        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        indices_clamped = indices.clamp(min=1, max=self.codebook_size - 2)

        cm = codebook[indices_clamped - 1]
        cc = codebook[indices_clamped]
        cp = codebook[indices_clamped + 1]

        remainder_index_m = (((cc - cm) * (x - cm)).sum(dim=1) / (cc - cm).square()
                             .sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        remainder_index_p = (((cp - cc) * (x - cc)).sum(dim=1) / (cp - cc).square()
                             .sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        z_q_m = ((1 - remainder_index_m) * cm) + (remainder_index_m * cc)
        z_q_p = ((1 - remainder_index_p) * cc) + (remainder_index_p * cp)
        distance_m = (x - z_q_m).square().sum(dim=1)
        distance_p = (x - z_q_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int64).squeeze() - 1

        c1 = codebook[indices_clamped + offset]
        c2 = codebook[indices_clamped + offset + 1]

        remainder_index = (((c2 - c1) * (x - c1)).sum(dim=1) / (c2 - c1).square()
                           .sum(dim=1)).clamp(min=0., max=1.)
        x_quantized = c1 + (remainder_index.reshape(-1, 1) * (c2 - c1))
        remainder = x - x_quantized
        return x_quantized, remainder, indices, perplexity


    def _replace_unused_entries(self) -> None:
        with torch.no_grad():

            for i in range(self.num_quantizers):

                usage_ratio = self.codebook_usage[i] / self.replacement_iters
                unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
                used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

                if unused_indices.numel() == 0 or used_indices.numel() == 0:
                    self.codebook_usage[i].zero_()
                    continue

                unused_count = unused_indices.numel()
                used_probs = self.codebook_usage[i, used_indices] / torch.sum(
                                                self.codebook_usage[i, used_indices])
                randomly_sampled_indices = used_probs.multinomial(
                                            num_samples=unused_count, replacement=True)
                sampled_indices = used_indices[randomly_sampled_indices]
                used_codebooks = self.codebook[i, sampled_indices].clone()

                self.codebook[i, unused_indices] = (used_codebooks + self.perturb_eps
                                            * torch.randn_like(used_codebooks)).clone()
                self.codebook_usage[i].zero_()

                if self.verbose:
                    print("\n***** Replaced " + str(unused_count) + f" codewords of"
                                                                    f" CB{i + 1} *****")

    def extra_repr(self) -> str:
        return (
            f"codebook_size={self.codebook_size}, "
            f"dim={self.dim}, "
            f"num_quantizers={self.num_quantizers}, "
            f"noise_var={self.noise_var}, "
            f"skip_iters={self.skip_iters}, "
            f"avg_iters={self.avg_iters}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}, "
            f"latents_on_cpu={self.latents_on_cpu}, "
            f"channel_first={self.channel_first}"
        )