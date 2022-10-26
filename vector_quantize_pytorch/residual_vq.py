from functools import partial

import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        heads = 1,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'

        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        codebooks = rearrange(codebooks, 'q 1 c d -> q c d')
        return codebooks

    def forward(
        self,
        x,
        return_all_codes = False
    ):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers

            codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = x.shape[0])
            gather_indices = repeat(all_indices, 'b n q -> q b n d', d = codebooks.shape[-1])

            all_codes = codebooks.gather(2, gather_indices) # gather all codes

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret
