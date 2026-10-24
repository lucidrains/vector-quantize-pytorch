from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, Tensor, is_tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from torch_einops_utils import pack_with_inverse, temp_eval

from vector_quantize_pytorch.lookup_free_quantization import LFQ

# constants

Return = namedtuple('Return', ['reconstructed', 'indices', 'entropy_aux_loss'])
Result = namedtuple('Result', ['pop_bits', 'best_gene', 'best_fitness', 'best_decoded'])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class EvoLFQ(Module):
    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        lfq: LFQ | None = None,
        *,
        dim: int | None = None,
        codebook_size: int | None = None,
        num_codebooks: int = 1,
        pop_size: int = 64,
        mutation_rate: float = 0.02,
        tournament_size: int = 2,
        elitism_count: int = 1,
        generations: int = 50,
        **lfq_kwargs
    ):
        super().__init__()
        assert pop_size > elitism_count, 'pop_size must be greater than elitism_count'

        self.encoder = encoder
        self.decoder = decoder

        if not exists(lfq):
            assert exists(dim) or exists(codebook_size), 'either lfq instance or dim / codebook_size must be supplied to EvoLFQ'
            lfq = LFQ(
                dim = dim,
                codebook_size = codebook_size,
                num_codebooks = num_codebooks,
                **lfq_kwargs
            )

        self.lfq = lfq
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.generations = generations

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def forward(self, x, **kwargs):
        latents = self.encoder(x)
        is_2d = latents.ndim == 2

        if is_2d:
            latents = rearrange(latents, 'b d -> b 1 d')

        quantized, indices, aux_loss = self.lfq(latents, **kwargs)

        if is_2d:
            quantized = rearrange(quantized, 'b 1 d -> b d')
            indices = rearrange(indices, 'b 1 ... -> b ...')

        reconstructed = self.decoder(quantized)
        return Return(reconstructed, indices, aux_loss)

    @torch.no_grad()
    def encode(self, x, return_signs = False):
        with temp_eval(self):
            latents = self.encoder(x)
            is_2d = latents.ndim == 2

            if is_2d:
                latents = rearrange(latents, 'b d -> b 1 d')

            quantized, indices, _ = self.lfq(latents)

            if is_2d:
                quantized = rearrange(quantized, 'b 1 d -> b d')

            if return_signs:
                return torch.where(quantized > 0, 1.0, -1.0)

            return (quantized > 0).float()

    @torch.no_grad()
    def decode_bits(self, bits):
        """
        Converts binary bits (0/1 or -1/+1) to codebook indices and uses lfq.indices_to_codes
        to decode back to data space.
        """
        with temp_eval(self):
            bits = (bits > 0).int()
            bits, inverse = pack_with_inverse(bits, '* d')

            codebook_dim = self.lfq.codebook_dim
            num_codebooks = self.lfq.num_codebooks

            if num_codebooks > 1 and bits.shape[-1] == codebook_dim * num_codebooks:
                bits = rearrange(bits, 'b (c d) -> b c d', c = num_codebooks, d = codebook_dim)

            if bits.ndim == 2:
                bits = rearrange(bits, 'b d -> b 1 d')

            mask = 2 ** torch.arange(codebook_dim - 1, -1, -1, device = self.device)
            indices = reduce(bits * mask, '... d -> ...', 'sum')

            if not self.lfq.keep_num_codebooks_dim and indices.ndim >= 2 and indices.shape[-1] == 1:
                indices = rearrange(indices, '... 1 -> ...')

            codes = self.lfq.indices_to_codes(indices)

            if codes.ndim == 3 and codes.shape[1] == 1:
                codes = rearrange(codes, 'b 1 d -> b d')

            codes = inverse(codes, '* d')
            return self.decoder(codes)

    # genetic algorithm helpers

    def init_random_population(self, pop_size = None, shape = None, device = None, is_sign = False):
        pop_size = default(pop_size, self.pop_size)
        device = default(device, self.device)
        bits = torch.randint(0, 2, (pop_size, *shape), device = device).float()

        if is_sign:
            return torch.where(bits > 0, 1.0, -1.0)

        return bits

    def init_population_from_data(self, x, pop_size = None, mutation_rate = None, is_sign = False):
        pop_size = default(pop_size, self.pop_size)
        mutation_rate = default(mutation_rate, self.mutation_rate)
        bits = self.encode(x, return_signs = is_sign)

        num_samples, *_ = bits.shape

        if num_samples < pop_size:
            repeats = (pop_size + num_samples - 1) // num_samples
            bits = repeat(bits, 'b ... -> (r b) ...', r = repeats)[:pop_size]
        else:
            bits = bits[:pop_size]

        if mutation_rate > 0:
            bits = self.mutate(bits, mutation_rate = mutation_rate, is_sign = is_sign)

        return bits

    def uniform_crossover(self, parent1, parent2):
        crossover_mask = torch.rand_like(parent1.float()) < 0.5
        return torch.where(crossover_mask, parent1, parent2)

    def mutate(self, population, mutation_rate = None, is_sign = False):
        mutation_rate = default(mutation_rate, self.mutation_rate)
        flip_mask = torch.rand_like(population.float()) < mutation_rate

        if is_sign:
            return torch.where(flip_mask, -population, population)

        return torch.where(flip_mask, 1.0 - population, population)

    def tournament_selection(self, population, fitnesses, tournament_size = None):
        tournament_size = default(tournament_size, self.tournament_size)
        pop_size, *_ = population.shape

        contenders = torch.randint(0, pop_size, (pop_size, tournament_size), device = self.device)
        winner_indices = contenders.gather(1, fitnesses[contenders].argmax(dim = -1, keepdim = True))
        return population[rearrange(winner_indices, 'b 1 -> b')]

    @torch.no_grad()
    def step(
        self,
        pop_bits,
        fitness_fn,
        tournament_size = None,
        mutation_rate = None,
        elitism_count = None,
        is_sign = False,
        batch_size = None,
        **kwargs
    ):
        tournament_size = default(tournament_size, self.tournament_size)
        mutation_rate = default(mutation_rate, self.mutation_rate)
        elitism_count = default(elitism_count, self.elitism_count)

        pop_size, *_ = pop_bits.shape
        num_offspring = pop_size - elitism_count

        # evaluate fitness

        if exists(batch_size):
            decoded_chunks = [self.decode_bits(chunk) for chunk in pop_bits.split(batch_size)]
            decoded = torch.cat(decoded_chunks, dim = 0)
        else:
            decoded = self.decode_bits(pop_bits)

        fitnesses = fitness_fn(decoded, pop_bits)
        if not is_tensor(fitnesses):
            fitnesses = torch.tensor(fitnesses, device = self.device, dtype = torch.float32)

        # sort by fitness descending and preserve elites

        sorted_indices = torch.argsort(fitnesses, descending = True)
        elites = pop_bits[sorted_indices[:elitism_count]].clone()

        # tournament selection: 2 parent tournaments for each of the num_offspring children needed

        contenders = torch.randint(0, pop_size, (num_offspring, 2, tournament_size), device = self.device)
        winner_rel_idx = fitnesses[contenders].argmax(dim = -1, keepdim = True)
        parent_indices = rearrange(contenders.gather(-1, winner_rel_idx), 'n p 1 -> n p')

        # crossover: 1 child produced per 2 parents

        p1, p2 = rearrange(pop_bits[parent_indices], 'n p ... -> p n ...')
        offspring = self.uniform_crossover(p1, p2)

        # mutation all at once

        offspring = self.mutate(offspring, mutation_rate = mutation_rate, is_sign = is_sign)

        next_pop = torch.cat([elites, offspring], dim = 0)
        return next_pop, fitnesses

    @torch.no_grad()
    def evolve(
        self,
        fitness_fn,
        pop_bits = None,
        pop_size = None,
        shape = None,
        generations = None,
        is_sign = False,
        return_best_decoded = False,
        **step_kwargs
    ):
        pop_size = default(pop_size, self.pop_size)
        generations = default(generations, self.generations)

        if not exists(pop_bits):
            assert exists(shape), 'shape must be provided if pop_bits is not supplied to evolve()'
            pop_bits = self.init_random_population(pop_size, shape, is_sign = is_sign)

        best_fitness = float('-inf')
        best_gene = None

        for _ in range(generations):
            evaluated_pop_bits = pop_bits
            pop_bits, fitnesses = self.step(
                pop_bits,
                fitness_fn,
                is_sign = is_sign,
                **step_kwargs
            )

            max_fit_idx = torch.argmax(fitnesses)
            max_fit = fitnesses[max_fit_idx].item()

            if max_fit > best_fitness:
                best_fitness = max_fit
                best_gene = evaluated_pop_bits[max_fit_idx].clone()

            best_decoded = None
            if return_best_decoded:
                best_decoded = self.decode_bits(rearrange(best_gene, '... -> 1 ...'))
                best_decoded = rearrange(best_decoded, '1 ... -> ...')

            yield Result(pop_bits, best_gene, best_fitness, best_decoded)
