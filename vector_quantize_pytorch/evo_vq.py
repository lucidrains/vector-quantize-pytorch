import torch
from torch import cat
from torch.nn import Module

from einops import reduce

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# evolution - start with the most minimal, a population of 3
# 1 is natural selected out, the other 2 performs crossover

def select_and_crossover(
    codes,   # Float[3 ...]
    fitness, # Float[3]
):
    pop_size = codes.shape[0]
    assert pop_size == fitness.shape[0]
    assert divisible_by(pop_size, 3)

    # selection

    sorted_indices = fitness.sort().indices
    selected = sorted_indices[(pop_size // 3):] # bottom third wins darwin awards
    codes = codes[selected]

    # crossover

    child = reduce(codes, '(two paired) ... -> paired ...', 'mean', two = 2)
    codes = cat((codes, child))

    return codes

# class

class EvoVQ(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
