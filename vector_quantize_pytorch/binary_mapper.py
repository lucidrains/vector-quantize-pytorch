from __future__ import annotations

# proposed in https://arxiv.org/abs/2510.17558 as a more stable alternative to VAE by François Fleuret

from math import log

import torch
from torch import nn, tensor, arange
import torch.nn.functional as F
from torch.nn import Module

from einops import einsum, pack, unpack

# constants

NAT = log(2)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def binary_entropy(logits):
    prob = logits.sigmoid()
    not_prob = 1. - prob
    return -(prob * F.logsigmoid(logits) + not_prob * F.logsigmoid(-logits)).sum(dim = -1)

def pack_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

# binary mapper

class BinaryMapper(Module):
    def __init__(
        self,
        bits = 1,
        kl_loss_threshold = NAT # 1 bit
    ):
        super().__init__()

        self.bits = bits
        self.num_codes = 2 ** bits

        power_two = 2 ** arange(bits)
        codes = (arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).byte().bool()

        self.register_buffer('power_two', power_two, persistent = False)
        self.register_buffer('codes', codes, persistent = False)

        # aux loss

        self.kl_loss_threshold = kl_loss_threshold
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        logits,
        temperature = 1.,
        straight_through = None,
        calc_aux_loss = None,
        return_indices = False
    ):
        straight_through = default(straight_through, self.training)
        calc_aux_loss = default(calc_aux_loss, self.training)

        assert logits.shape[-1] == self.bits, f'logits must have a last dimension of {self.bits}'

        # allow for any number of leading dimensions

        logits, inverse_pack_lead_dims = pack_with_inverse(logits, '* bits')

        # temperature and prob for sampling

        prob_for_sample = (logits / temperature).sigmoid()

        # sampling

        sampled_bits = (torch.rand_like(logits) <= prob_for_sample).long()
        indices = (self.power_two * sampled_bits).sum(dim = -1)

        one_hot = F.one_hot(indices, self.num_codes).float()

        # maybe calculate aux loss

        aux_kl_loss = self.zero

        if calc_aux_loss:
            # calculate negative entropy

            kl_div = self.bits * NAT - binary_entropy(logits)
            aux_kl_loss = F.relu(kl_div - self.kl_loss_threshold).mean()

        # maybe straight through

        if straight_through:
            # get the soft G for the gradients and do a straight through

            soft_G = (
                einsum(F.logsigmoid(logits), self.codes.float(), '... bits, codes bits -> ... codes') +
                einsum(F.logsigmoid(-logits), (~self.codes).float(), '... bits, codes bits -> ... codes')
            ).exp()

            # straight through

            one_hot = one_hot + soft_G - soft_G.detach()

        # inverse pack

        one_hot = inverse_pack_lead_dims(one_hot)
        indices = inverse_pack_lead_dims(indices, '*')

        # returning

        if not return_indices:
            return one_hot, aux_kl_loss

        # also allow for returning indices, even though it can be derived from sparse output with an argmax

        return one_hot, indices, aux_kl_loss

# allow for quick copy paste

if __name__ == '__main__':

    binary_mapper = BinaryMapper(bits = 8)

    logits = torch.randn(3, 4, 8)

    sparse_one_hot, indices, aux_loss = binary_mapper(logits, return_indices = True)

    assert sparse_one_hot.shape == (3, 4, 2 ** 8)
    assert indices.shape == (3, 4)
    assert aux_loss.numel() == 1
