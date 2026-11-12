import pytest
import torch
from torch.nn import functional as F

from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import gumbel_sample


@pytest.mark.parametrize('topk', (1, 2, 4))
@pytest.mark.parametrize('temperature', (.4, 1., 2.))
def test_topk_straight_through_samples_share_the_softmax_surrogate(topk, temperature):
    torch.manual_seed(5)
    logits = torch.randn(2, 5, 4, requires_grad = True)
    expected_logits = logits.detach().clone().requires_grad_()

    indices, samples = gumbel_sample(
        logits, temperature = temperature, straight_through = True, topk = topk
    )

    expected_indices = logits.detach().topk(topk, dim = -1).indices
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(samples, F.one_hot(expected_indices, 4).float())
    weights = torch.randn_like(samples)
    (samples * weights).sum().backward()
    expected = (expected_logits / temperature).softmax(-1).unsqueeze(-2)
    (expected * weights).sum().backward()
    assert expected_logits.grad.abs().sum() > 0
    torch.testing.assert_close(logits.grad, expected_logits.grad)


@pytest.mark.parametrize('training, temperature', ((False, 1.), (True, 0.)))
def test_topk_without_soft_surrogate_keeps_hard_samples(training, temperature):
    logits = torch.tensor([[[1., 3., 2.], [4., 0., 5.]]], requires_grad = True)
    indices, samples = gumbel_sample(
        logits, training = training, temperature = temperature,
        straight_through = True, topk = 2
    )
    expected = logits.detach().topk(2, dim = -1).indices
    torch.testing.assert_close(indices, expected)
    torch.testing.assert_close(samples, F.one_hot(expected, 3).float())
    assert not samples.requires_grad


def test_vector_quantize_topk_gumbel_backward_matches_soft_assignment_reference():
    torch.manual_seed(13)
    quantizer = VectorQuantize(
        dim = 2, codebook_size = 4, freeze_codebook = True,
        straight_through = True, rotation_trick = False,
        route_gradients_to_input = False, commitment_weight = 0.,
        sample_codebook_temp = .7
    )
    codes = torch.tensor([[-2., -1.], [-1., 2.], [2., 1.], [1., -2.]])
    quantizer.codebook = codes
    values = torch.randn(2, 3, 2, requires_grad = True)
    reference_values = values.detach().clone().requires_grad_()

    actual, indices, _ = quantizer(values, topk = 2)

    torch.testing.assert_close(actual, codes[indices])
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    probabilities = (-torch.cdist(reference_values, codes) / .7).softmax(-1)
    soft_codes = probabilities @ codes
    (soft_codes.unsqueeze(-2) * weights).sum().backward()
    assert reference_values.grad.abs().sum() > 0
    torch.testing.assert_close(values.grad, reference_values.grad)
