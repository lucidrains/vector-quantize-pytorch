from copy import deepcopy

import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize('heads, separate', ((1, False), (2, False), (2, True)))
@pytest.mark.parametrize('explicit_mask', (False, True))
def test_ignored_indices_do_not_enter_codebook_statistics(heads, separate, explicit_mask):
    torch.manual_seed(0)
    quantizer = VectorQuantize(
        dim = 2 * heads, codebook_dim = 2, heads = heads,
        separate_codebook_per_head = separate, codebook_size = 2,
        decay = .5, dead_code_threshold_ratio = 0.
    )
    values = torch.tensor([[[1., 2., 3., 4.], [20., 30., 40., 50.], [5., 6., 7., 8.]]])
    indices = torch.tensor([[[0, -1], [-1, 1], [1, 0]]])
    values = values[..., :2 * heads]
    indices = indices[..., :heads]
    if heads == 1:
        indices = indices.squeeze(-1)
    mask = torch.tensor([[True, True, False]]) if explicit_mask else None
    old_count = quantizer._codebook.cluster_size.clone()
    old_sum = quantizer._codebook.embed_avg.clone()
    expected_count = torch.zeros_like(old_count)
    expected_sum = torch.zeros_like(old_sum)

    for token in range(3):
        if explicit_mask and not mask[0, token]:
            continue
        for head in range(heads):
            index = indices[0, token] if heads == 1 else indices[0, token, head]
            if index == -1:
                continue
            book = head if separate else 0
            expected_count[book, index] += 1
            expected_sum[book, index] += values[0, token, 2 * head:2 * head + 2]

    quantizer.update_indices(values, indices, mask = mask)

    torch.testing.assert_close(quantizer._codebook.cluster_size, .5 * (old_count + expected_count))
    torch.testing.assert_close(quantizer._codebook.embed_avg, .5 * (old_sum + expected_sum))


@pytest.mark.parametrize('alias', (False, True))
def test_reusing_masked_forward_indices_matches_explicit_update_mask(alias):
    torch.manual_seed(4)
    source = VectorQuantize(dim = 2, codebook_size = 4, decay = .8).eval()
    values = torch.randn(2, 5, 2)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    _, indices, _ = source(values, mask = mask)
    actual = deepcopy(source).train()
    reference = deepcopy(source).train()

    update = actual.update_ema_indices if alias else actual.update_indices
    update(values, indices)
    reference.update_indices(values, indices, mask = mask)

    for name in ('cluster_size', 'embed_avg', 'embed'):
        torch.testing.assert_close(getattr(actual._codebook, name), getattr(reference._codebook, name))


def test_fully_ignored_indices_do_not_initialize_code_zero_statistics():
    quantizer = VectorQuantize(dim = 2, codebook_size = 4, dead_code_threshold_ratio = .1)
    initial = deepcopy(quantizer.state_dict())

    quantizer.update_indices(torch.ones(1, 3, 2), torch.full((1, 3), -1))

    for name, expected in initial.items():
        torch.testing.assert_close(quantizer.state_dict()[name], expected)
