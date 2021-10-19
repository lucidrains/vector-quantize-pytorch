import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def kmeans(x, num_clusters, num_iters = 10, use_cosine_sim = False):
    samples = rearrange(x, '... d -> (...) d')
    num_samples, dim, dtype, device = *samples.shape, x.dtype, x.device

    if num_samples >= num_clusters:
        indices = torch.randperm(num_samples, device=device)[:num_clusters]
    else:
        indices = torch.randint(0, num_samples, (num_clusters,), device=device)

    means = samples[indices]

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
            buckets = dists.max(dim = -1).indices
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = (diffs ** 2).sum(dim = -1)
            buckets = dists.argmin(dim = -1)

        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins[..., None]

        if use_cosine_sim:
            new_means = F.normalize(new_means, dim = -1)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, x):
        if not self.initted:
            self.init_embed_(x)

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim = -1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(x.dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = F.normalize(torch.randn(codebook_size, dim), dim = -1)
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)

    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim = True)
        self.embed.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = F.normalize(flatten, dim = -1)
        embed = F.normalize(self.embed, dim = - 1)

        if not self.initted:
            self.init_embed_(flatten)

        dist = flatten @ embed.t()
        embed_ind = dist.max(dim = -1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(0)
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, dim = -1)
            embed_normalized = torch.where(zero_mask[..., None], embed, embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)

        return quantize, embed_ind

# main class

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed = None,
        codebook_dim = None,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
        kmeans_init = False,
        kmeans_iters = 10,
        use_cosine_sim = False
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment = commitment

        klass = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = klass(
            dim = codebook_dim,
            codebook_size = n_embed,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps
        )

    @property
    def codebook(self):
        return self._codebook.codebook

    def forward(self, x):
        dtype = x.dtype
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        commit_loss = 0.
        if self.training:
            commit_loss = F.mse_loss(quantize.detach(), x) * self.commitment
            quantize = x + (quantize - x).detach()

        quantize = self.project_out(quantize)
        return quantize, embed_ind, commit_loss
