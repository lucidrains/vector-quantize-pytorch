import math

import pytest
import torch
from torch import nn
from vector_quantize_pytorch import FSP
from vector_quantize_pytorch.finite_scalar_perturbation import build_cdf_act


def scalar_cdf(value):
    return 0.5 * math.exp(value) if value < 0 else 1 - 0.5 * math.exp(-value)


def scalar_inverse(probability):
    return (
        math.log(2 * probability)
        if probability < 0.5
        else -math.log(2 * (1 - probability))
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inverse", [False, True])
def test_laplace_pair_has_the_analytic_midpoint_derivative(dtype, inverse):
    cdf, icdf = build_cdf_act("laplace")
    if inverse:
        values = [0.01, 0.25, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 0.75, 0.99]
        inputs = torch.tensor(values, dtype=dtype, requires_grad=True)
        actual = icdf(inputs)
        represented = inputs.detach().tolist()
        expected = torch.tensor([scalar_inverse(p) for p in represented], dtype=dtype)
        expected_gradient = torch.tensor(
            [1 / min(p, 1 - p) for p in represented], dtype=dtype
        )
    else:
        values = [-1000, -1, -1e-6, 0, 1e-6, 1, 1000]
        inputs = torch.tensor(values, dtype=dtype, requires_grad=True)
        actual = cdf(inputs)
        represented = inputs.detach().tolist()
        expected = torch.tensor([scalar_cdf(z) for z in represented], dtype=dtype)
        expected_gradient = torch.tensor(
            [0.5 * math.exp(-abs(z)) for z in represented], dtype=dtype
        )
    gradient = torch.autograd.grad(actual.sum(), inputs)[0]
    assert torch.isfinite(actual).all() and torch.isfinite(gradient).all()
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        gradient, expected_gradient, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("mode", ["train_perturb", "train_quantize", "eval"])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_complete_fsp_trains_central_features_with_analytic_reconstruction_gradients(
    mode, projected, channel_first, inverse
):
    torch.manual_seed(293)
    dim = 3 if projected else 2
    encoder = nn.Linear(dim, dim, bias=False)
    model = FSP(
        levels=[3, 5],
        dim=dim,
        channel_first=channel_first,
        act_name="laplace",
        need_inv_act=inverse,
        vector_norm="var_laplace",
        quantize_rate=1.0 if mode == "train_quantize" else 0.0,
    ).train(mode != "eval")
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(dim))
        if projected:
            model.project_in.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
            model.project_in.bias.zero_()
            model.project_out.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
            )
            model.project_out.bias.zero_()
    rows = torch.tensor(
        [
            [[0.0, 0.0, 0.2], [0.3, -0.2, -0.3], [-0.4, 0.2, 0.5], [0.6, -0.5, 0.1]],
            [[0.0, 0.1, -0.2], [-0.2, 0.0, 0.3], [0.5, -0.3, -0.4], [-0.3, 0.6, 0.2]],
        ]
    )[..., :dim].requires_grad_()
    encoded = encoder(rows)
    native_input = encoded.movedim(-1, 1) if channel_first else encoded
    torch.manual_seed(307)
    output, indices, norm_loss, _ = model(native_input)
    latents = encoded[..., :2].detach().reshape(-1, 2)
    probabilities = torch.tensor(
        [[scalar_cdf(float(z)) for z in row] for row in latents]
    )
    levels = torch.tensor([3, 5])
    level_indices = (probabilities * levels).floor()
    expected_probabilities = (level_indices + 0.5) / levels
    if mode == "train_perturb":
        torch.manual_seed(307)
        noise = (torch.rand_like(probabilities) * 2 - 1) / (2 * levels)
        proposals = probabilities + noise
        accepted = (proposals > 0) & (proposals < 1)
        expected_probabilities = torch.where(accepted, proposals, probabilities)
        # Native FSP also samples the perturb/quantize selector at rate zero.
        selectors = torch.rand_like(probabilities) > 0
        expected_probabilities = torch.where(
            selectors, expected_probabilities, (level_indices + 0.5) / levels
        )
    if inverse:
        expected_latent_output = torch.tensor(
            [[scalar_inverse(float(p)) for p in row] for row in expected_probabilities]
        )
        latent_derivative = torch.ones_like(latents)
    else:
        expected_latent_output = (expected_probabilities - 0.5) / 0.28867513459481287
        latent_derivative = torch.tensor(
            [
                [0.5 * math.exp(-abs(float(z))) / 0.28867513459481287 for z in row]
                for row in latents
            ]
        )
    expected_output = expected_latent_output
    output_weight = torch.eye(2)
    input_weight = torch.eye(2)
    if projected:
        output_weight = model.project_out.weight.detach()
        input_weight = model.project_in.weight.detach()
        expected_output = nn.functional.linear(
            expected_output, output_weight, model.project_out.bias.detach()
        )
    expected_output = expected_output.reshape(rows.shape)
    if channel_first:
        expected_output = expected_output.movedim(-1, 1)
    torch.testing.assert_close(output, expected_output, rtol=2e-5, atol=2e-6)
    expected_indices = (level_indices * torch.tensor([1, 3])).sum(-1).to(torch.int32)
    assert torch.equal(indices, expected_indices.reshape(rows.shape[:-1]))
    if mode != "train_perturb":
        torch.testing.assert_close(model.indices_to_codes(indices), output)

    # Differentiate the density and linear maps analytically. The inverse mode
    # deliberately uses native FSP's identity straight-through Jacobian.
    latent_gradient = latent_derivative * output_weight.sum(0)
    expected_input_gradient = (latent_gradient @ input_weight).reshape(rows.shape)
    actual_input_gradient = torch.autograd.grad(output.sum(), rows, retain_graph=True)[
        0
    ]
    torch.testing.assert_close(actual_input_gradient, expected_input_gradient)
    expected_encoder_gradient = expected_input_gradient.flatten(
        0, 1
    ).t() @ rows.detach().flatten(0, 1)
    actual_encoder_gradient = torch.autograd.grad(
        output.sum(), encoder.weight, retain_graph=True
    )[0]
    torch.testing.assert_close(actual_encoder_gradient, expected_encoder_gradient)
    norm_gradient = torch.autograd.grad(norm_loss, encoder.weight, retain_graph=True)[0]
    expected_total_gradient = expected_encoder_gradient + 0.01 * norm_gradient
    original_weight = encoder.weight.detach().clone()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    (output.sum() + 0.01 * norm_loss).backward()
    optimizer.step()
    torch.testing.assert_close(
        encoder.weight, original_weight - 0.01 * expected_total_gradient
    )
    assert torch.isfinite(rows.grad).all()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
