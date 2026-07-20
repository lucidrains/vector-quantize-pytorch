import pytest
import torch

from vector_quantize_pytorch import FSP
from vector_quantize_pytorch.finite_scalar_perturbation import build_cdf_act

# CDF activation roundtrip


@pytest.mark.parametrize("act_name", ("tanh", "sigmoid", "normal", "laplace", "cauchy"))
def test_cdf_act_roundtrip(act_name):
    act_func, inv_act_func = build_cdf_act(act_name)

    x = torch.randn(64, 10)
    y = act_func(x)
    x_hat = inv_act_func(y)
    assert (y > 0.0).all() and (y < 1.0).all()
    assert torch.allclose(x, x_hat, atol=1e-4), (
        f"{act_name} roundtrip error: {(x_hat - x).abs().max()}"
    )


# FSP


def test_fsp_basic():
    fsp = FSP(levels=[8, 5, 5, 5], act_name="normal", vector_norm="none")

    x = torch.randn(1, 1024, 4)
    quantized, indices, norm_loss, other_info = fsp(x)
    assert quantized.shape == x.shape
    assert indices.shape == (1, 1024)
    assert norm_loss.item() == 0.0
    assert isinstance(other_info, dict)


def test_fsp_eval_roundtrip():
    fsp = FSP(levels=[8, 5, 5, 5])
    fsp.eval()

    x = torch.randn(1, 1024, 4)
    quantized, indices, *_ = fsp(x)
    recovered = fsp.indices_to_codes(indices)
    assert torch.allclose(quantized, recovered, atol=1e-5), (
        f"max diff: {(quantized - recovered).abs().max()}"
    )


def test_fsp_index_encoding():
    fsp = FSP(levels=[8, 5, 5, 5])

    # Test known level_indices (max values for each dimension)
    level_indices = torch.tensor([[[7, 4, 4, 4]]])  # shape: (1, 1, 4)
    flat_index = fsp.level_indices_to_indices(level_indices)
    # Expected: 7*(5*5*5) + 4*(5*5) + 4*5 + 4 = 875 + 100 + 20 + 4 = 999
    assert flat_index.item() == 999

    # Verify decoding recovers original level_indices
    recovered = fsp.indices_to_level_indices(flat_index)
    assert torch.equal(level_indices, recovered)

    # Test zero indices (boundary case)
    level_indices_zero = torch.tensor([[[0, 0, 0, 0]]])
    flat_index_zero = fsp.level_indices_to_indices(level_indices_zero)
    assert flat_index_zero.item() == 0
    recovered_zero = fsp.indices_to_level_indices(flat_index_zero)
    assert torch.equal(level_indices_zero, recovered_zero)


def test_fsp_quantize_rate_one():
    fsp = FSP(levels=[8, 5, 5, 5], quantize_rate=1.0)
    fsp.train()

    x = torch.randn(1, 64, 4)
    out1, *_ = fsp(x)
    out2, *_ = fsp(x)
    assert torch.allclose(out1, out2)


def test_fsp_image_input():
    fsp = FSP(levels=[8, 5, 5, 5], dim=4, channel_first=True)
    fsp.eval()

    x = torch.randn(2, 4, 8, 8)
    quantized, indices, *_ = fsp(x)
    assert quantized.shape == x.shape
    assert indices.shape == (2, 8, 8)

    recovered = fsp.indices_to_codes(indices)
    assert recovered.shape == x.shape
    assert torch.allclose(quantized, recovered, atol=1e-5)


def test_fsp_with_dim_projection():
    """dim != codebook_dim requires projection layers."""
    fsp = FSP(levels=[8, 5, 5, 5], dim=256)
    fsp.eval()

    assert fsp.has_projections

    x = torch.randn(1, 64, 256)
    quantized, indices, _, _ = fsp(x)
    assert quantized.shape == x.shape
    assert indices.shape == (1, 64)

    recovered = fsp.indices_to_codes(indices)
    assert recovered.shape == x.shape
    assert torch.allclose(quantized, recovered, atol=1e-4)


@pytest.mark.parametrize("dtype,use_autocast", [
    (torch.float32, False),
    (torch.float64, False),
    (torch.float16, True),
    (torch.bfloat16, True),
])
def test_fsp_training(dtype, use_autocast):
    """Test FSP with different precisions, autocast, and gradient flow."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        FSP(levels=[8, 5, 5, 5], dim=256),
        torch.nn.Linear(256, 256),
    ).to(device).to(dtype)
    model.train()

    x = torch.randn(2, 64, 256, dtype=dtype, device=device, requires_grad=True)
    # Forward pass with or without autocast
    if use_autocast:
        with torch.amp.autocast(device, dtype=dtype):
            h1 = model[0](x)
            quantized, indices, norm_loss, other_info = model[1](h1)
            out = model[2](quantized)
    else:
        h1 = model[0](x)
        quantized, indices, norm_loss, other_info = model[1](h1)
        out = model[2](quantized)

    # Verify output dtype matches input
    assert quantized.dtype == dtype
    assert indices.dtype == torch.int32

    # Verify indices are valid (no NaN/Inf from precision issues)
    assert not indices.isnan().any()
    assert (indices >= 0).all() and (indices < model[1].codebook_size).all()

    # Backward pass - verify gradient flow
    loss = out.sum() + norm_loss
    loss.backward()

    assert x.grad is not None, "Input gradient should exist"
    assert not x.grad.isnan().any()
    assert not x.grad.isinf().any()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.isnan().any()
        assert not parameter.grad.isinf().any()
