import torch
from torch import nn
import torch.nn.functional as F
import pytest

from vector_quantize_pytorch import LFQ, EvoLFQ

def test_evo_lfq_2d_data():
    """
    Verifies EvoLFQ on 2D data inputs: (batch, dim)
    """
    dim = 16
    codebook_size = 256  # 8 bits

    encoder = nn.Sequential(
        nn.Linear(dim, 32),
        nn.ReLU(),
        nn.Linear(32, 8)
    )

    decoder = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, dim)
    )

    evo_lfq = EvoLFQ(
        encoder = encoder,
        decoder = decoder,
        codebook_size = codebook_size,
        pop_size = 16
    )

    # 2D data input: (batch, dim)
    x = torch.randn(8, dim)

    # 1. Forward pass with 2D data
    reconstructed, indices, aux_loss = evo_lfq(x)
    assert reconstructed.shape == (8, dim), f"Expected (8, 16), got {reconstructed.shape}"
    assert indices.shape == (8,), f"Expected (8,), got {indices.shape}"

    # 2. Encode 2D data to binary bits
    bits = evo_lfq.encode(x)
    assert bits.shape == (8, 8), f"Expected (8, 8), got {bits.shape}"
    assert ((bits == 0) | (bits == 1)).all()

    # 3. Decode bits back to 2D data space
    x_hat = evo_lfq.decode_bits(bits)
    assert x_hat.shape == (8, dim), f"Expected (8, 16), got {x_hat.shape}"

    # 4. Evolution with 2D target
    target = torch.randn(dim)

    def fitness_fn(decoded_batch, bits_batch):
        return -((decoded_batch - target) ** 2).mean(dim = -1)

    for result in evo_lfq.evolve(
        fitness_fn = fitness_fn,
        shape = (8,),
        generations = 5,
        return_best_decoded = True
    ):
        pass

    assert result.pop_bits.shape == (16, 8)
    assert result.best_gene.shape == (8,)
    assert result.best_decoded.shape == (dim,)

def test_evo_lfq_2d_data_multi_codebook():
    """
    Verifies EvoLFQ on 2D data with multiple codebooks.
    """
    dim = 32
    num_codebooks = 4
    codebook_size = 256  # 8 bits per codebook -> 32 bits total

    encoder = nn.Linear(dim, 32)
    decoder = nn.Linear(32, dim)

    evo_lfq = EvoLFQ(
        encoder = encoder,
        decoder = decoder,
        codebook_size = codebook_size,
        num_codebooks = num_codebooks,
        pop_size = 16
    )

    x = torch.randn(8, dim)

    reconstructed, indices, aux_loss = evo_lfq(x)
    assert reconstructed.shape == (8, dim)
    assert indices.shape == (8, num_codebooks)

    bits = evo_lfq.encode(x)
    assert bits.shape == (8, 32)

    x_hat = evo_lfq.decode_bits(bits)
    assert x_hat.shape == (8, dim)

def test_evo_lfq_3d_data():
    """
    Verifies EvoLFQ on 3D data inputs: (batch, seq_len, dim)
    """
    dim = 16
    seq_len = 10
    codebook_size = 256  # 8 bits

    encoder = nn.Linear(dim, 8)
    decoder = nn.Linear(8, dim)

    evo_lfq = EvoLFQ(
        encoder = encoder,
        decoder = decoder,
        codebook_size = codebook_size,
        pop_size = 16
    )

    x = torch.randn(4, seq_len, dim)

    reconstructed, indices, aux_loss = evo_lfq(x)
    assert reconstructed.shape == (4, seq_len, dim)
    assert indices.shape == (4, seq_len)

    bits = evo_lfq.encode(x)
    assert bits.shape == (4, seq_len, 8)

    x_hat = evo_lfq.decode_bits(bits)
    assert x_hat.shape == (4, seq_len, dim)

def test_ga_operators():
    codebook_size = 16  # 4 bits
    dim = 8

    encoder = nn.Linear(dim, 4)
    decoder = nn.Linear(4, dim)

    evo_lfq = EvoLFQ(
        encoder = encoder,
        decoder = decoder,
        codebook_size = codebook_size,
        pop_size = 16
    )

    pop = evo_lfq.init_random_population(16, (4,))
    assert pop.shape == (16, 4)

    p1 = torch.ones(4)
    p2 = torch.zeros(4)
    c = evo_lfq.uniform_crossover(p1, p2)
    assert c.shape == (4,)
    assert ((c == 0.0) | (c == 1.0)).all()

    mutated = evo_lfq.mutate(p1, mutation_rate = 1.0)
    assert (mutated == 0.0).all()

    fitnesses = torch.arange(16, dtype = torch.float32)
    selected = evo_lfq.tournament_selection(pop, fitnesses, tournament_size = 4)
    assert selected.shape == (16, 4)

def test_evolution_resuming():
    dim = 8
    encoder = nn.Linear(dim, 8)
    decoder = nn.Linear(8, dim)

    evo_lfq = EvoLFQ(
        encoder = encoder,
        decoder = decoder,
        codebook_size = 256,
        pop_size = 32
    )

    target = torch.randn(dim)

    def fitness_fn(decoded_batch, bits_batch):
        return -((decoded_batch - target) ** 2).mean(dim = -1)

    for result in evo_lfq.evolve(fitness_fn = fitness_fn, shape = (8,), generations = 5):
        pass

    # Resume with pop_bits
    for result_resumed in evo_lfq.evolve(fitness_fn = fitness_fn, pop_bits = result.pop_bits, generations = 5):
        pass

    assert result_resumed.pop_bits.shape == (32, 8)
    assert result_resumed.best_fitness >= result.best_fitness
