import pytest
import torch
from torch import nn

from vector_quantize_pytorch import LFQ, EvoLFQ


@pytest.mark.parametrize("is_sign", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("generations", [1, 3])
def test_best_gene_and_decoded_output_match_evaluated_fitness(
    is_sign, batch_size, generations
):
    model = EvoLFQ(
        nn.Identity(),
        nn.Identity(),
        lfq=LFQ(dim=3, codebook_size=8),
        pop_size=8,
        elitism_count=7,
        mutation_rate=1.0,
    )
    # The maximum initially occupies row four. Elite sorting moves it to row
    # zero, making old-fitness/new-population index mismatches deterministic.
    values = torch.tensor([0, 1, 2, 3, 7, 4, 5, 6])
    weights = torch.tensor([4, 2, 1])
    bits = ((values[:, None] & weights) != 0).float()
    population = bits * 2 - 1 if is_sign else bits
    evaluated = []

    def fitness(decoded, genes):
        expected_bits = (genes > 0).float()
        torch.testing.assert_close(decoded, expected_bits * 2 - 1)
        scores = (expected_bits * weights).sum(-1)
        evaluated.append((genes.clone(), scores.clone()))
        return scores

    results = model.evolve(
        fitness,
        pop_bits=population,
        generations=generations,
        is_sign=is_sign,
        batch_size=batch_size,
        return_best_decoded=True,
    )
    for generation, result in enumerate(results, 1):
        assert len(evaluated) == generation
        expected_score = max(scores.max().item() for _, scores in evaluated)
        assert result.best_fitness == expected_score == weights.sum().item()
        expected_gene = torch.ones(3)
        torch.testing.assert_close(result.best_gene, expected_gene)
        torch.testing.assert_close(result.best_decoded, torch.ones(3))
        actual_score = ((result.best_gene > 0).float() * weights).sum().item()
        assert actual_score == result.best_fitness
        assert result.pop_bits.shape == population.shape


def test_best_gene_is_preserved_when_returned_population_is_modified():
    model = EvoLFQ(
        nn.Identity(),
        nn.Identity(),
        lfq=LFQ(dim=2, codebook_size=4),
        pop_size=4,
        elitism_count=3,
    )
    population = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    def fitness(decoded, genes):
        return (genes * torch.tensor([2.0, 1.0])).sum(-1)

    result = next(model.evolve(fitness, pop_bits=population, generations=1))
    result.pop_bits.fill_(0)
    torch.testing.assert_close(result.best_gene, torch.ones(2))
    assert result.best_decoded is None
