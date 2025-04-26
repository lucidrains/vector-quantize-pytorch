import torch
from torch import cat

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# evolution - start with the most minimal, a population of 3
# 1 is natural selected out, the other 2 performs crossover

def select_and_crossover(
    codes,   # Float[3 ...]
    fitness, # Float[3]
):
    assert codes.shape[0] == fitness.shape[0] == 3

    # selection

    top2 = fitness.topk(2, dim = -1).indices
    codes = codes[top2]

    # crossover

    child = codes.mean(dim = 0, keepdim = True)
    codes = cat((codes, child))

    return codes

# class

class EvoVQ(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
