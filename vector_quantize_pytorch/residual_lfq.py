import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch.lookup_free_quantization import LFQ

from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class ResidualLFQ(nn.Module):
    raise NotImplementedError
