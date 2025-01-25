import torch
from torch import nn
from torch.nn import Module, ModuleList

# quantization

from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.residual_vq import ResidualVQ, GroupedResidualVQ
from vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer
from vector_quantize_pytorch.finite_scalar_quantization import FSQ
from vector_quantize_pytorch.lookup_free_quantization import LFQ
from vector_quantize_pytorch.residual_lfq import ResidualLFQ, GroupedResidualLFQ
from vector_quantize_pytorch.residual_fsq import ResidualFSQ, GroupedResidualFSQ
from vector_quantize_pytorch.latent_quantization import LatentQuantize
from vector_quantize_pytorch.sim_vq import SimVQ
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ

QUANTIZE_KLASSES = (
    VectorQuantize,
    ResidualVQ,
    GroupedResidualVQ,
    RandomProjectionQuantizer,
    FSQ,
    LFQ,
    SimVQ,
    ResidualSimVQ,
    ResidualLFQ,
    GroupedResidualLFQ,
    ResidualFSQ,
    GroupedResidualFSQ,
    LatentQuantize
)

# classes

class Sequential(Module):
    def __init__(
        self,
        *fns: Module
    ):
        super().__init__()
        assert sum([int(isinstance(fn, QUANTIZE_KLASSES)) for fn in fns]) == 1, 'this special Sequential must contain exactly one quantizer'

        self.fns = ModuleList(fns)

    def forward(
        self,
        x,
        **kwargs
    ):
        for fn in self.fns:

            if not isinstance(fn, QUANTIZE_KLASSES):
                x = fn(x)
                continue

            x, *rest = fn(x, **kwargs)

        output = (x, *rest)

        return output
