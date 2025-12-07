import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.quasirandom import SobolEngine
from scipy.stats import norm
import math

def prior_samples(n_samples, n_variable, seed_rec):
    sobol = SobolEngine(n_variable, scramble=True, seed=seed_rec)
    samples_sobol = sobol.draw(n_samples)
    samples_i = torch.from_numpy(norm.ppf(samples_sobol))
    return samples_i

class GaussianQuant(nn.Module):
    def __init__(self, dim, dim_idx, codebook_size, logvar_range=[-30.0, 20.0], # common parameters
                 tolerance = 0.5, lam_factor=1.01, # train parameters
                 seed=42, beta=1.0, use_ste=True):
        super().__init__()
        self.dim_idx = dim_idx
        self.dim = dim
        self.logvar_range = logvar_range
        self.n_samples = codebook_size
        self.log_n_samples = int(math.log(self.n_samples, 2))
        # training parameter setup
        self.lam_factor = lam_factor
        self.lam = 1.0
        self.lam_min = 1.0
        self.lam_max = 1.0
        self.lam_range = (1e-7, 1e7)
        self.tolerance = tolerance

        # inference parameter setup
        self.beta = beta
        self.seed = seed
        self.register_buffer("prior_samples", prior_samples(self.n_samples, self.dim, self.seed).float(), persistent=False)
        self.normal_dist = Normal(torch.zeros([1, self.dim]), torch.ones([1, self.dim]))
        self.register_buffer("normal_log_prob", self.normal_dist.log_prob(self.prior_samples).float(), persistent=False)

        self.logvar_range = logvar_range
        self.use_ste = use_ste
        self.perturbed = None

    def quant_gaussian(self, z):
        z = torch.movedim(z, self.dim_idx, -1)
        assert(z.shape[-1] % (self.dim * 2) == 0)
        z_shape = z.shape
        z = z.reshape(-1, z_shape[-1])
        codebook_num=z.shape[-1] // (self.dim * 2)
        # get \mu and \sigma 
        mu, logvar = z.chunk(2, -1)
        logvar = torch.clamp(logvar, self.logvar_range[0], self.logvar_range[1])
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)

        # Gaussian VAE
        zhat = mu + torch.randn_like(mu) * std
        # kl divergence in log 2
        kl2 = 1.4426 * 0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)
        # -1, dim, codebook number
        kl2 = kl2.reshape(-1,self.dim,codebook_num)
        kl2 = torch.sum(kl2,dim=1) # sum over dim

        # compute mean, min, max of kl divergence
        kl2_mean, kl2_min, kl2_max = torch.mean(kl2), torch.min(kl2), torch.max(kl2)
        ge = (kl2 > self.log_n_samples + self.tolerance).type(kl2.dtype) * self.lam_max
        eq = (kl2 <= self.log_n_samples + self.tolerance).type(kl2.dtype) * (
            kl2 >= self.log_n_samples - self.tolerance
        ).type(kl2.dtype)
        le = (kl2 < self.log_n_samples - self.tolerance).type(kl2.dtype) * self.lam_min

        # reweight kl divergence according to its relation to log2 codebook_size
        kl_loss = ge * kl2 + eq * kl2 + le * kl2
        kl_loss = torch.mean(kl_loss) * self.lam
        # update lambda
        if kl2_mean > self.log_n_samples:
            self.lam = self.lam * self.lam_factor
        else:
            self.lam = self.lam / self.lam_factor

        if kl2_max > self.log_n_samples + self.tolerance:
            self.lam_max = self.lam_max * self.lam_factor
        else:
            self.lam_max / self.lam_max * self.lam_factor
        self.lam_max = max(min(self.lam_max, self.lam_range[1]), 1.0)
        if kl2_min < self.log_n_samples - self.tolerance:
            self.lam_min = self.lam_min / self.lam_factor
        else:
            self.lam_min = self.lam_min * self.lam_factor
        self.lam_min = max(min(self.lam_min, 1.0), self.lam_range[0])

        zhat = zhat.reshape(*z_shape[:-1], -1)
        zhat = torch.movedim(zhat, -1, self.dim_idx)

        mu = mu.reshape(*z_shape[:-1], -1)
        mu = torch.movedim(mu, -1, self.dim_idx)

        std = std.reshape(*z_shape[:-1], -1)
        std = torch.movedim(std, -1, self.dim_idx)

        info = {"kl-loss": torch.mean(kl_loss), "bits-mean": kl2_mean, "bits-min": kl2_min, "bits-max": kl2_max, 
                "lam-min": self.lam_min, "lam-max": self.lam_max, "lam": self.lam, "mu": mu, "std": std, "zhat_noquant": zhat}

        return zhat, info

    def quant_vq(self, z):

        # reshape to (-1, dim * codebook_num)
        z = torch.movedim(z, self.dim_idx, -1)
        assert(z.shape[-1] % (self.dim * 2) == 0)
        z_shape = z.shape
        z = z.reshape(-1, z_shape[-1])
        codebook_num=z.shape[-1] // (self.dim * 2)
        # get \mu and \sigma 
        mu, logvar = z.chunk(2, -1)
        logvar = torch.clamp(logvar, self.logvar_range[0], self.logvar_range[1])
        std = torch.exp(0.5 * logvar)

        # reshape everything into (-1, dim)
        mu = mu.reshape(-1,self.dim,codebook_num).permute(0,2,1).reshape(-1,self.dim)
        std = std.reshape(-1,self.dim,codebook_num).permute(0,2,1).reshape(-1,self.dim)

        # process per batch to avoid OOM
        bs = mu.shape[0] // 8
        zhat = torch.zeros_like(mu)
        indices = torch.zeros([mu.shape[0]], device=mu.device, dtype=torch.long)
        for i in range(0, mu.shape[0], bs):
            mu_q = mu[i:i+bs]
            std_q = std[i:i+bs]
            q_normal_dist = Normal(mu_q[:, None, :], std_q[:, None, :])
            log_ratios = (
                q_normal_dist.log_prob(self.prior_samples[None])
                - self.normal_log_prob[None] * self.beta
            )
            perturbed = torch.sum(log_ratios, dim=2)
            argmax_indices = torch.argmax(perturbed, dim=1)
            zhat[i:i+bs] = torch.index_select(self.prior_samples, 0, argmax_indices)
            indices[i:i+bs] = argmax_indices
        zhat = zhat.reshape(-1,codebook_num,self.dim).permute(0,2,1).reshape(-1,codebook_num*self.dim).float()
        indices = indices.reshape(-1,codebook_num)

        zhat = zhat.reshape(*z_shape[:-1], -1)
        zhat = torch.movedim(zhat, -1, self.dim_idx)

        indices = indices.reshape(*z_shape[:-1], -1)
        indices = torch.movedim(indices, -1, self.dim_idx)

        info = {"indices": indices, "zhat_quant": zhat}

        return zhat, info

    def forward(self, z):
        zhat_g, info_g = self.quant_gaussian(z)
        with torch.no_grad():
            zhat_v, info_v = self.quant_vq(z)
        if self.use_ste:
            zhat = zhat_g - zhat_g.detach() + zhat_v
        else:
            if self.training:
                zhat = zhat_g
            else:
                zhat = zhat_v
        info = info_g | info_v
        return zhat, info

    def indices_to_codes(
        self,
        indices
    ):
        indices = torch.movedim(indices, self.dim_idx, -1)
        i_shape = indices.shape
        codebook_num = i_shape[-1]
        indices = indices.reshape(-1)
        zhat = torch.zeros([indices.shape[0], self.dim], device=indices.device, dtype=torch.float32)
        zhat = torch.index_select(self.prior_samples, 0, indices).float()
        zhat = zhat.reshape(-1,codebook_num,self.dim).permute(0,2,1).reshape(-1,codebook_num*self.dim)

        # back to original shape
        zhat = zhat.reshape(*i_shape[:-1], -1)
        zhat = torch.movedim(zhat, -1, self.dim_idx)
        return zhat


if __name__ == "__main__":
    mu = torch.zeros([1,6,64,64])
    logvar = torch.zeros([1,6,64,64])
    input = torch.cat([mu, logvar],dim=1)
    gq = GaussianQuant(dim=3, dim_idx=1, codebook_size=128)
    gq.train()
    zhat, log = gq(input)
    loss = log["kl-loss"]

    gq.eval()
    zhat_vq, log = gq(input)
    indices = log["indices"]
    zhat2 = gq.indices_to_codes(indices)
    print(torch.sum(torch.abs(zhat_vq - zhat2)))
    print(torch.mean(torch.abs(zhat_vq - mu)))
    print(torch.mean(torch.abs(log["zhat_noquant"] - mu)))