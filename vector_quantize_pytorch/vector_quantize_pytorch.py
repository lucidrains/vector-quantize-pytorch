from __future__ import annotations

from functools import partial, cache
from collections import namedtuple

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.amp import autocast

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def identity(t):
    return t

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return None
    elif len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def entropy(prob, eps = 1e-5):
    return (-prob * log(prob, eps = eps)).sum(dim = -1)

def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith('mps:')

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def lens_to_mask(lens, max_length):
    seq = torch.arange(max_length, device = lens.device)
    return seq < lens[:, None]

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(dim = dim)
    one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    assert total_count == 0, f'invalid total count {total_count}'

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

# rotation trick related

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )

def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)

# distributed helpers

@cache
def is_distributed():
    return distributed.is_initialized() and distributed.get_world_size() > 1

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

# distance types

class EuclideanCodebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        manual_ema_update = False,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update
        self.manual_ema_update = manual_ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask = None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased = False)

        # calculate codebook mean and variance

        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.training:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')

            self.embed.data[ind][mask] = sampled
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def update_ema(self):
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False,
        codebook_transform_fn: Callable | None = None
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, unpack_one = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)

        # affine params

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        # get maybe learnable codes

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # handle maybe implicit neural codebook
        # and calculate distance

        if exists(codebook_transform_fn):
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            broadcastable_input = rearrange(flatten, '... d -> ... 1 d')

            dist = -F.pairwise_distance(broadcastable_input, transformed_embed)
        else:
            dist = -cdist(flatten, embed)

        # sample or argmax depending on temperature

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)

        embed_ind = unpack_one(embed_ind, 'h *')

        if exists(codebook_transform_fn):
            transformed_embed = unpack_one(transformed_embed, 'h * c d')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')

            if exists(codebook_transform_fn):
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

        else:
            if exists(codebook_transform_fn):
                quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
            else:
                quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)

        if self.training and self.ema_update and not freeze_codebook:

            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.

            cluster_size = embed_onehot.sum(dim = 1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            if not self.manual_ema_update:
                self.update_ema()
                self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, 'h * d')

        return quantize, embed_ind, dist

class CosineSimCodebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        manual_ema_update = False
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.manual_ema_update = manual_ema_update

        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')

            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def update_ema(self):
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        embed_normalized = l2norm(embed_normalized)

        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False,
        codebook_transform_fn: Callable | None = None
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype

        flatten, unpack_one = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # handle maybe implicit neural codebook
        # and compute cosine sim distance

        if exists(codebook_transform_fn):
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            transformed_embed = l2norm(transformed_embed)

            dist = einsum('h n d, h n c d -> h n c', flatten, transformed_embed)
        else:
            dist = einsum('h n d, h c d -> h n c', flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, 'h *')

        if exists(codebook_transform_fn):
            transformed_embed = unpack_one(transformed_embed, 'h * c d')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')

            if exists(codebook_transform_fn):
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

        else:
            if exists(codebook_transform_fn):
                quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
            else:
                quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)

        if self.training and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot[~mask] = 0.

            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            if not self.manual_ema_update:
                self.update_ema()
                self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, 'h * d')
        return quantize, embed_ind, dist

# main class

LossBreakdown = namedtuple('LossBreakdown', [
    'commitment',
    'codebook_diversity',
    'orthogonal_reg',
    'inplace_optimize',
])

class VectorQuantize(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        layernorm_after_project_in = False, # proposed by @SaltyChtao here https://github.com/lucidrains/vector-quantize-pytorch/issues/26#issuecomment-1324711561
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        codebook_diversity_loss_weight = 0.,
        codebook_diversity_temperature = 100.,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        rotation_trick = True,  # Propagate grads through VQ layer w/ rotation trick: https://arxiv.org/abs/2410.06424 by @cfifty
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        manual_ema_update = False,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        manual_in_place_optimizer_update = False,
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0., # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        return_zeros_for_masked_padding = True
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim

        self.project_in = Sequential(
            nn.Linear(dim, codebook_input_dim),
            nn.LayerNorm(codebook_input_dim) if layernorm_after_project_in else None
        ) if requires_projection else nn.Identity()

        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps

        self.has_commitment_loss = commitment_weight > 0.
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss

        assert not (use_cosine_sim and learnable_codebook), 'cosine sim distance codebook not compatible with learnable codebook yet'
        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0.
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        has_codebook_diversity_loss = codebook_diversity_loss_weight > 0.
        self.has_codebook_diversity_loss = has_codebook_diversity_loss
        self.codebook_diversity_temperature = codebook_diversity_temperature
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight

        assert not (straight_through and rotation_trick)
        self.rotation_trick = rotation_trick

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        self.sync_update_v = sync_update_v

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            straight_through = straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = is_distributed()

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update,
            manual_ema_update = manual_ema_update
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self.use_cosine_sim = use_cosine_sim
        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None
        self.manual_in_place_optimizer_update = manual_in_place_optimizer_update

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # for variable lengthed sequences, whether to take care of masking out the padding to 0 (or return the original input)
        self.return_zeros_for_masked_padding = return_zeros_for_masked_padding

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
        else:
            indices, unpack_one = pack_one(indices, 'b * h')
            indices = rearrange(indices, 'b n h -> b h n')

            indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
            codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

            codes = codebook.gather(2, indices)
            codes = rearrange(codes, 'b h n d -> b n (h d)')
            codes = unpack_one(codes, 'b * d')

        if not self.channel_last:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def update_in_place_optimizer(self):
        if not exists(self.in_place_codebook_optimizer):
            return

        self.in_place_codebook_optimizer.step()
        self.in_place_codebook_optimizer.zero_grad()

    def maybe_split_heads_from_input(self, x):
        if self.heads == 1:
            return x

        ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
        return rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = self.heads)

    def expire_codes_(self, x):
        x = self._codebook.transform_input(x)
        x = self.maybe_split_heads_from_input(x)
        self._codebook.expire_codes_(x)

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        lens = None,
        sample_codebook_temp = None,
        freeze_codebook = False,
        return_loss_breakdown = False,
        codebook_transform_fn: Callable | None = None
    ):
        orig_input, input_requires_grad = x, x.requires_grad

        # handle masking, either passed in as `mask` or `lens`

        assert not (exists(mask) and exists(lens))

        if exists(lens):
            mask = lens_to_mask(lens, x.shape[1])

        # handle one token given

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            assert not exists(mask)
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        x = self.maybe_split_heads_from_input(x)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook,
            codebook_transform_fn = codebook_transform_fn
        )

        # quantize

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # losses for loss breakdown

        commit_loss = orthogonal_reg_loss = inplace_optimize_loss = codebook_diversity_loss = self.zero

        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()

            if not self.manual_in_place_optimizer_update:
                self.update_in_place_optimizer()

            inplace_optimize_loss = loss

            # quantize again

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)

            # spare rotation trick calculation if inputs do not need gradients

            if input_requires_grad:
                if self.rotation_trick:
                    quantize = rotate_to(x, quantize)
                else:
                    # standard STE to get gradients through VQ layer.
                    quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            # calculate codebook diversity loss (negative of entropy) if needed

            if self.has_codebook_diversity_loss:
                prob = (-distances * self.codebook_diversity_temperature).softmax(dim = -1)
                avg_prob = reduce(prob, '... n l -> n l', 'mean')
                codebook_diversity_loss = -entropy(avg_prob).mean()

                loss = loss + codebook_diversity_loss * self.codebook_diversity_loss_weight

            # commitment loss

            if self.has_commitment_loss:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True

        if exists(mask):
            masked_out_value = orig_input

            if self.return_zeros_for_masked_padding:
                masked_out_value = torch.zeros_like(orig_input)

            quantize = einx.where(
                'b n, b n d, b n d -> b n d',
                mask,
                quantize,
                masked_out_value
            )

            embed_ind = einx.where(
                'b n, b n ..., -> b n ...',
                mask,
                embed_ind,
                -1
            )

        if not return_loss_breakdown:
            return quantize, embed_ind, loss

        loss_breakdown = LossBreakdown(commit_loss, codebook_diversity_loss, orthogonal_reg_loss, inplace_optimize_loss)

        return quantize, embed_ind, loss, loss_breakdown
