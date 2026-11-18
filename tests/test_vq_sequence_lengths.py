from copy import deepcopy

import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize('channel_last', (False, True))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('heads, separate', ((1, False), (2, False), (2, True)))
@pytest.mark.parametrize('zero_padding', (False, True))
def test_lengths_match_explicit_masks_for_each_sequence_layout(
    channel_last, training, heads, separate, zero_padding
):
    torch.manual_seed(7)
    quantizer = VectorQuantize(
        dim = 4,
        codebook_dim = 4 // heads,
        heads = heads,
        separate_codebook_per_head = separate,
        codebook_size = 8,
        channel_last = channel_last,
        rotation_trick = False,
        freeze_codebook = True,
        commitment_weight = .1 if heads == 1 else 0.,
        return_zeros_for_masked_padding = zero_padding
    ).train(training)
    reference = deepcopy(quantizer)
    values = torch.randn(3, 7, 4)
    values = values if channel_last else values.transpose(1, 2)
    values = values.detach().requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    lengths = torch.tensor([7, 4, 1])
    mask = torch.arange(7)[None, :] < lengths[:, None]

    expected, expected_indices, expected_loss = reference(reference_values, mask = mask)
    actual, actual_indices, actual_loss = quantizer(values, lens = lengths)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_loss, expected_loss)
    assert (actual_indices[~mask] == -1).all()
    actual_tokens = actual if channel_last else actual.transpose(1, 2)
    input_tokens = values if channel_last else values.transpose(1, 2)
    padding = torch.zeros_like(input_tokens[~mask]) if zero_padding else input_tokens[~mask]
    torch.testing.assert_close(actual_tokens[~mask], padding)

    if training:
        (actual.square().sum() + actual_loss.sum()).backward()
        (expected.square().sum() + expected_loss.sum()).backward()
        torch.testing.assert_close(values.grad, reference_values.grad)


@pytest.mark.parametrize('channel_last', (False, True))
def test_sequence_lengths_match_explicit_masks_with_topk_codes(channel_last):
    torch.manual_seed(11)
    quantizer = VectorQuantize(
        dim = 3,
        codebook_size = 8,
        channel_last = channel_last,
        freeze_codebook = True
    ).eval()
    values = torch.randn(2, 5, 3)
    values = values if channel_last else values.transpose(1, 2)
    lengths = torch.tensor([5, 2])
    mask = torch.arange(5)[None, :] < lengths[:, None]

    expected = quantizer(values, mask = mask, topk = 3)
    actual = quantizer(values, lens = lengths, topk = 3)

    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
