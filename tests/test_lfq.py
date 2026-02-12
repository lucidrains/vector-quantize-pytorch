import torch
import pytest
import math
from vector_quantize_pytorch import LFQ

# helpers

def exists(val):
    return val is not None

# tests

@pytest.mark.parametrize('frac_per_sample_entropy', (1., 0.5))
@pytest.mark.parametrize('mask', (
    torch.tensor([False, False]),
    torch.tensor([True, False]),
    torch.tensor([True, True])
))
def test_masked_lfq(
    frac_per_sample_entropy,
    mask
):
    quantizer = LFQ(
        codebook_size = 65536,
        dim = 16,
        entropy_loss_weight = 0.1,
        diversity_gamma = 1.,
        frac_per_sample_entropy = frac_per_sample_entropy
    )

    image_feats = torch.randn(2, 16, 32, 32)

    ret, _ = quantizer(image_feats, inv_temperature = 100., return_loss_breakdown = True, mask = mask)

    quantized, indices, _ = ret
    assert (quantized == quantizer.indices_to_codes(indices)).all()

@pytest.mark.parametrize('frac_per_sample_entropy', (0.1,))
@pytest.mark.parametrize('iters', (10,))
@pytest.mark.parametrize('mask', (None, torch.tensor([True, False])))
def test_lfq_bruteforce_frac_per_sample_entropy(
    frac_per_sample_entropy,
    iters,
    mask
):
    image_feats = torch.randn(2, 16, 32, 32)

    full_per_sample_entropy_quantizer = LFQ(
        codebook_size = 65536,
        dim = 16,
        entropy_loss_weight = 0.1,
        diversity_gamma = 1.,
        frac_per_sample_entropy = 1
    )

    partial_per_sample_entropy_quantizer = LFQ(
        codebook_size = 65536,
        dim = 16,
        entropy_loss_weight = 0.1,
        diversity_gamma = 1.,
        frac_per_sample_entropy = frac_per_sample_entropy
    )

    ret, loss_breakdown = full_per_sample_entropy_quantizer(image_feats, inv_temperature = 100., return_loss_breakdown = True, mask = mask)
    true_per_sample_entropy = loss_breakdown.per_sample_entropy

    per_sample_losses = torch.zeros(iters)

    for i in range(iters):
        ret, loss_breakdown = partial_per_sample_entropy_quantizer(image_feats, inv_temperature = 100., return_loss_breakdown = True, mask = mask)

        quantized, indices, _ = ret
        assert (quantized == partial_per_sample_entropy_quantizer.indices_to_codes(indices)).all()
        per_sample_losses[i] = loss_breakdown.per_sample_entropy

    # 95% confidence interval
    assert abs(per_sample_losses.mean() - true_per_sample_entropy) < (1.96 * (per_sample_losses.std() / math.sqrt(iters)))