import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize('heads, separate', ((1, False), (2, False), (2, True)))
@pytest.mark.parametrize('channel_last', (False, True))
@pytest.mark.parametrize('projected', (False, True))
@pytest.mark.parametrize('masked', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_supervised_outputs_match_normal_quantization(heads, separate, channel_last, projected, masked, training):
    torch.manual_seed(31)
    quantizer = VectorQuantize(
        dim = 4, codebook_dim = 1 if projected else 4 // heads,
        heads = heads, separate_codebook_per_head = separate,
        codebook_size = 8, channel_last = channel_last,
        commitment_weight = 0., freeze_codebook = True, rotation_trick = False
    ).train(training)
    values = torch.randn(2, 5, 4)
    values = values if channel_last else values.transpose(1, 2)
    values = values.detach().requires_grad_()
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, True]]) if masked else None
    expected, indices, _ = quantizer(values, mask = mask)

    actual, loss = quantizer(values, indices = indices, mask = mask)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(loss)
    if training:
        targets = (values, *quantizer.project_out.parameters())
        actual_gradients = torch.autograd.grad(actual.square().sum(), targets, retain_graph = True)
        expected_gradients = torch.autograd.grad(expected.square().sum(), targets)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('channel_last', (False, True))
def test_supervised_topk_preserves_projection_and_layout(channel_last):
    torch.manual_seed(19)
    quantizer = VectorQuantize(
        dim = 4, codebook_dim = 2, codebook_size = 8,
        channel_last = channel_last, commitment_weight = 0.,
        freeze_codebook = True, rotation_trick = False
    )
    values = torch.randn(2, 5, 4)
    values = values if channel_last else values.transpose(1, 2)
    expected, candidates, _ = quantizer(values, topk = 2)

    actual, loss = quantizer(values, indices = candidates[..., 0], topk = 2)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(loss)
