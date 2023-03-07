import torch
from torch import nn, einsum
import torch.nn.functional as F
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, pack, unpack

class RandomProjectionQuantizer(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        codebook_dim,
        num_codebooks = 1,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)

        self.register_buffer('rand_projs', rand_projs)

        self.vq = VectorQuantize(
            dim = codebook_dim * num_codebooks,
            heads = num_codebooks,
            codebook_size = codebook_size,
            use_cosine_sim = True,
            separate_codebook_per_head = True,
            **kwargs
        )

    @torch.no_grad()
    def forward(self, x):

        x = einsum('b n d, h d e -> b n h e', x, self.rand_projs)
        x, ps = pack([x], 'b n *')

        self.vq.eval()
        _, indices, _ = self.vq(x)

        return indices
