"""Saved shared-codebook indices must reconstruct the concatenated quantizer heads."""

import pytest
import torch
from torch import nn

from vector_quantize_pytorch import VectorQuantize

CONFIGURATIONS = ((1, False), (2, False), (3, False), (2, True), (3, True))


def make_quantizer(heads, separate, projected, dtype, *, learnable=False):
    codebook_dim = 3
    dim = 5 if projected else heads * codebook_dim
    quantizer = VectorQuantize(
        dim=dim,
        codebook_dim=codebook_dim,
        heads=heads,
        separate_codebook_per_head=separate,
        codebook_size=4,
        ema_update=not learnable,
        learnable_codebook=learnable,
        rotation_trick=False,
        route_gradients_to_input=not learnable,
    ).to(dtype=dtype)
    return quantizer


def reference_codes(quantizer, indices):
    codebook = quantizer.codebook
    if quantizer.heads == 1:
        return codebook[indices]
    if quantizer.separate_codebook_per_head:
        selected = [
            codebook[head][indices[..., head]] for head in range(quantizer.heads)
        ]
    else:
        selected = [codebook[indices[..., head]] for head in range(quantizer.heads)]
    return torch.cat(selected, dim=-1)


@pytest.mark.parametrize("heads,separate", CONFIGURATIONS)
@pytest.mark.parametrize("projected", (False, True))
@pytest.mark.parametrize("single_token", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("training", (False, True))
def test_native_forward_indices_reconstruct_shared_and_separate_heads(
    *, heads, separate, projected, single_token, dtype, training
):
    torch.manual_seed(130)
    quantizer = make_quantizer(heads, separate, projected, dtype)
    quantizer.train(training)
    shape = (3, quantizer.dim) if single_token else (2, 5, quantizer.dim)
    x = torch.randn(shape, dtype=dtype, requires_grad=training)
    before = quantizer.codebook.detach().clone()
    # Compare within one unchanged codebook state. Training EMA intentionally
    # updates its state; freeze_codebook is the supported forward control for it.
    quantized, indices, loss = quantizer(x, freeze_codebook=True)
    expected_index_shape = shape[:-1] + ((heads,) if heads > 1 else ())
    assert indices.shape == expected_index_shape
    assert indices.dtype == torch.long and indices.device == x.device
    expected_codes = reference_codes(quantizer, indices)
    actual_codes = quantizer.get_codes_from_indices(indices)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    assert actual_codes.shape == shape[:-1] + (heads * 3,)
    assert actual_codes.dtype == dtype and actual_codes.device == x.device
    decoded = quantizer.get_output_from_indices(indices)
    assert decoded.shape == quantized.shape == x.shape
    torch.testing.assert_close(
        decoded, quantizer.project_out(expected_codes), rtol=0, atol=0
    )
    torch.testing.assert_close(decoded, quantized, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(quantizer.codebook, before, rtol=0, atol=0)
    if training:
        (quantized.square().mean() + loss).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("heads", (2, 3))
@pytest.mark.parametrize("projected", (False, True))
def test_saved_indices_preserve_head_order_and_decoder_gradients(heads, projected):
    torch.manual_seed(254)
    quantizer = make_quantizer(heads, False, projected, torch.float64, learnable=True)
    codebook = torch.tensor(
        [
            [0.25, 0.50, 0.75],
            [1.25, 1.50, 1.75],
            [2.25, 2.50, 2.75],
            [3.25, 3.50, 3.75],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        quantizer.codebook = codebook
    indices = torch.stack(
        [(torch.arange(8).reshape(2, 4) + head) % 4 for head in range(heads)], dim=-1
    )
    expected_codes = torch.cat(
        [codebook[indices[..., head]] for head in range(heads)], dim=-1
    )
    actual_codes = quantizer.get_codes_from_indices(indices)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    expected = reference_codes(quantizer, indices)
    if projected:
        expected = nn.functional.linear(
            expected, quantizer.project_out.weight, quantizer.project_out.bias
        )
    actual = quantizer.get_output_from_indices(indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = [quantizer._codebook.embed]
    if projected:
        parameters += [quantizer.project_out.weight, quantizer.project_out.bias]
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )
        assert torch.isfinite(actual_gradient).all()
