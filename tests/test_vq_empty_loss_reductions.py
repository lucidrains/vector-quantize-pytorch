import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize('cross_entropy', (False, True))
@pytest.mark.parametrize('inplace', (False, True))
@pytest.mark.parametrize('population', ('empty', 'partial', 'full'))
def test_masked_losses_remain_finite_without_valid_tokens(cross_entropy, inplace, population):
    torch.manual_seed(3)
    quantizer = VectorQuantize(
        dim = 2, codebook_size = 4, rotation_trick = False,
        commitment_use_cross_entropy_loss = cross_entropy,
        learnable_codebook = inplace, ema_update = not inplace,
        in_place_codebook_optimizer = (lambda params: torch.optim.SGD(params, lr = .1)) if inplace else None
    )
    values = torch.randn(2, 3, 2, requires_grad = True)
    mask = torch.ones(2, 3, dtype = torch.bool)
    if population == 'empty':
        mask.zero_()
    elif population == 'partial':
        mask[0, 1:] = False
    output, indices, loss, breakdown = quantizer(values, mask = mask, return_loss_breakdown = True)

    assert torch.isfinite(loss).all()
    assert torch.isfinite(breakdown.commitment).all()
    assert torch.isfinite(breakdown.inplace_optimize).all()
    if population == 'empty':
        assert loss.item() == 0
        assert breakdown.commitment.item() == 0
        assert breakdown.inplace_optimize.item() == 0
        torch.testing.assert_close(output, torch.zeros_like(output))
        assert (indices == -1).all()
    else:
        assert loss.item() > 0

    loss.sum().backward()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad[~mask], torch.zeros_like(values.grad[~mask]))
    for parameter in quantizer.parameters():
        assert torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_supervised_all_ignored_codes_have_zero_differentiable_loss():
    quantizer = VectorQuantize(dim = 2, codebook_size = 4, freeze_codebook = True)
    values = torch.randn(2, 3, 2, requires_grad = True)

    _, loss = quantizer(values, indices = torch.full((2, 3), -1))

    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
