import math

import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


def make_quantizer(heads = 1, separate = False):
    quantizer = VectorQuantize(
        dim = heads,
        codebook_dim = 1,
        heads = heads,
        separate_codebook_per_head = separate,
        codebook_size = 2,
        commitment_weight = 0.,
        rotation_trick = False,
        codebook_diversity_loss_weight = 1.,
        codebook_diversity_temperature = 1.,
        freeze_codebook = True
    )
    codes = torch.tensor([[-.5], [.5]])
    quantizer.codebook = codes.repeat(heads, 1, 1) if separate else codes
    return quantizer


def reference_loss(values, mask = None, separate = False):
    # Each head has the two scalar codewords -.5 and .5.
    values = values if mask is None else values[mask]
    values = values.reshape(-1, values.shape[-1])
    logits = -(values[..., None] - values.new_tensor([-.5, .5])).abs()
    probabilities = logits.softmax(-1)
    average = probabilities.mean(0) if separate else probabilities.mean((0, 1))
    return -torch.distributions.Categorical(probs = average).entropy().mean()


@pytest.mark.parametrize('batch, sequence', ((1, 4), (2, 2), (4, 1)))
def test_diversity_uses_all_tokens_independently_of_batch_partition(batch, sequence):
    values = torch.tensor([-3., -2., 2., 3.]).reshape(batch, sequence, 1).requires_grad_()
    quantizer = make_quantizer()

    _, _, loss = quantizer(values)

    torch.testing.assert_close(loss.squeeze(), torch.tensor(-math.log(2.)))
    expected = reference_loss(values)
    expected_gradient, = torch.autograd.grad(expected, values, retain_graph = True)
    actual_gradient, = torch.autograd.grad(loss.sum(), values)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('separate', (False, True))
def test_diversity_retains_independent_codebook_populations(separate):
    values = torch.tensor([[[-2., 2.], [-3., 3.]], [[-4., 4.], [-2., 2.]]])
    quantizer = make_quantizer(heads = 2, separate = separate)

    _, _, loss = quantizer(values)

    torch.testing.assert_close(loss.squeeze(), reference_loss(values, separate = separate))


@pytest.mark.parametrize('heads, separate', ((1, False), (2, False), (2, True)))
def test_padding_does_not_change_diversity_or_receive_its_gradient(heads, separate):
    values = torch.tensor([[[-.3], [.2], [7.]], [[-.2], [6.], [8.]]])
    values = values.repeat(1, 1, heads).requires_grad_()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    quantizer = make_quantizer(heads = heads, separate = separate)

    _, _, loss = quantizer(values, mask = mask)

    expected = reference_loss(values, mask = mask, separate = separate)
    torch.testing.assert_close(loss.squeeze(), expected)
    expected_gradient, = torch.autograd.grad(expected, values, retain_graph = True)
    actual_gradient, = torch.autograd.grad(loss.sum(), values)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert expected_gradient[mask].abs().sum() > 0
    torch.testing.assert_close(actual_gradient[~mask], torch.zeros_like(actual_gradient[~mask]))


def test_fully_masked_population_has_zero_diversity_loss_and_gradient():
    values = torch.tensor([[[-2.], [3.]]], requires_grad = True)
    quantizer = make_quantizer()

    _, _, loss = quantizer(values, mask = torch.zeros((1, 2), dtype = torch.bool))

    torch.testing.assert_close(loss.squeeze(), torch.tensor(0.))
    loss.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
