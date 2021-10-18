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

def kmeans(x, num_clusters, num_iters = 10):
    samples = rearrange(x, '... d -> (...) d')
    num_samples, dim, dtype, device = *samples.shape, x.dtype, x.device

    if num_samples >= num_clusters:
        indices = torch.randperm(num_samples, device=device)[:num_clusters]
    else:
        indices = torch.randint(0, num_samples, (num_clusters,), device=device)

    means = samples[indices]

    for _ in range(num_iters):
        diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
        dists = (diffs ** 2).sum(dim = -1)
        buckets = dists.argmin(dim = -1)

        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins[..., None]
        means = torch.where(zero_mask[..., None], means, new_means)

    return rearrange(means, 'n d -> d n')

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
        kmeans_init = False,
        kmeans_iters = 10
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(dim, n_embed)

        self.kmeans_iters = kmeans_iters
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def init_embed_(self, data):
        embed = kmeans(data, self.n_embed, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, input):
        if not self.initted:
            self.init_embed_(input)

        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])

        commit_loss = 0.
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            commit_loss = F.mse_loss(quantize.detach(), input) * self.commitment
            quantize = input + (quantize - input).detach()

        return quantize, embed_ind, commit_loss
