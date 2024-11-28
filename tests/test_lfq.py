import torch
import pytest
from vector_quantize_pytorch import LFQ
import math
"""
testing_strategy:
subdivisions: using masks, using frac_per_sample_entropy < 1
"""

torch.manual_seed(0)

@pytest.mark.parametrize('frac_per_sample_entropy', (1., 0.5))
@pytest.mark.parametrize('mask', (torch.tensor([False, False]),
                                  torch.tensor([True, False]),
                                  torch.tensor([True, True])))
def test_masked_lfq(
    frac_per_sample_entropy,
    mask
):
    # you can specify either dim or codebook_size
    # if both specified, will be validated against each other

    quantizer = LFQ(
        codebook_size = 65536,      # codebook size, must be a power of 2
        dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
        diversity_gamma = 1.,       # within entropy loss, how much weight to give to diversity 
        frac_per_sample_entropy = frac_per_sample_entropy
    )

    image_feats = torch.randn(2, 16, 32, 32)

    ret, loss_breakdown = quantizer(image_feats, inv_temperature=100., return_loss_breakdown=True, mask=mask)  # you may want to experiment with temperature

    quantized, indices, _ = ret
    assert (quantized == quantizer.indices_to_codes(indices)).all()

@pytest.mark.parametrize('frac_per_sample_entropy', (0.1,))
@pytest.mark.parametrize('iters', (10,))
@pytest.mark.parametrize('mask', (None, torch.tensor([True, False])))
def test_lfq_bruteforce_frac_per_sample_entropy(frac_per_sample_entropy, iters, mask):
    image_feats = torch.randn(2, 16, 32, 32)

    full_per_sample_entropy_quantizer = LFQ(
        codebook_size = 65536,      # codebook size, must be a power of 2
        dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
        diversity_gamma = 1.,       # within entropy loss, how much weight to give to diversity 
        frac_per_sample_entropy = 1
    )

    partial_per_sample_entropy_quantizer = LFQ(
        codebook_size = 65536,      # codebook size, must be a power of 2
        dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
        diversity_gamma = 1.,       # within entropy loss, how much weight to give to diversity 
        frac_per_sample_entropy = frac_per_sample_entropy
    )

    ret, loss_breakdown = full_per_sample_entropy_quantizer(
        image_feats, inv_temperature=100., return_loss_breakdown=True, mask=mask)
    true_per_sample_entropy = loss_breakdown.per_sample_entropy

    per_sample_losses = torch.zeros(iters)
    for iter in range(iters):
        ret, loss_breakdown = partial_per_sample_entropy_quantizer(
            image_feats, inv_temperature=100., return_loss_breakdown=True, mask=mask)  # you may want to experiment with temperature

        quantized, indices, _ = ret
        assert (quantized == partial_per_sample_entropy_quantizer.indices_to_codes(indices)).all()
        per_sample_losses[iter] = loss_breakdown.per_sample_entropy
    # 95% confidence interval
    assert abs(per_sample_losses.mean() - true_per_sample_entropy) \
        < (1.96*(per_sample_losses.std() / math.sqrt(iters)))

    print("difference: ", abs(per_sample_losses.mean() - true_per_sample_entropy))
    print("std error:", (1.96*(per_sample_losses.std() / math.sqrt(iters))))