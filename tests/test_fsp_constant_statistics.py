import pytest
import torch

from vector_quantize_pytorch import FSP
from vector_quantize_pytorch.finite_scalar_perturbation import VectorNorm


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize('preset', ('none', 'var_tanh', 'kurt'))
def test_vector_norm_constant_features_have_finite_gradients(dtype, preset):
    z = torch.ones(8, 4, dtype = dtype, requires_grad = True)
    norm = VectorNorm.build(preset)

    loss, stats = norm(z)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in stats.values())
    assert torch.isfinite(z.grad).all()
    expected_grad = torch.full_like(z, 2 * norm.l1_weight / z.numel())
    torch.testing.assert_close(z.grad, expected_grad)


@pytest.mark.parametrize('dtype', (torch.float16, torch.float32))
def test_fsp_constant_encoder_output_can_take_an_optimizer_step(dtype):
    torch.manual_seed(0)
    encoder = torch.nn.Linear(6, 4).to(dtype)
    torch.nn.init.zeros_(encoder.weight)
    torch.nn.init.zeros_(encoder.bias)
    optimizer = torch.optim.SGD(encoder.parameters(), lr = 0.01)
    fsp = FSP(levels = [8, 5, 5, 5], quantize_rate = 1.)
    x = torch.randn(8, 6, dtype = dtype)

    quantized, _, norm_loss, _ = fsp(encoder(x))
    loss = quantized.float().square().mean() + norm_loss
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.bias.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in encoder.parameters())
    assert encoder.bias.abs().sum() > 0


def test_vector_norm_preserves_nondegenerate_moments():
    z = torch.tensor([[-2., 0.], [1., 2.], [3., -1.], [0., 4.]], dtype = torch.float64)
    norm = VectorNorm.build('kurt')

    _, stats = norm(z)

    centered = z - z.mean(dim = 0)
    variance = centered.square().sum(dim = 0) / (z.shape[0] - 1)
    standardized = centered / variance.sqrt()
    expected = dict(
        mean = z.mean(dim = 0),
        variance = variance,
        skewness = standardized.pow(3).mean(dim = 0),
        kurtosis = standardized.pow(4).mean(dim = 0) - 3.
    )
    for name, value in expected.items():
        torch.testing.assert_close(stats[name], value)
