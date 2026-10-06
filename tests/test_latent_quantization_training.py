from functools import partial

import pytest
import torch
from torch import nn
from vector_quantize_pytorch import LatentQuantize


def latent_reference(model, inputs):
    rows = inputs.movedim(1, -1)
    shape = rows.shape
    rows = rows.reshape(inputs.shape[0], -1, model.dim)
    if model.has_projections:
        rows = nn.functional.linear(
            rows, model.project_in.weight, model.project_in.bias
        )
    latents = rows.reshape(*rows.shape[:2], model.num_codebooks, model.codebook_dim)
    assignments = []
    selected = []
    for dim, values in enumerate(model.values_per_latent):
        distances = (latents[..., dim, None].detach() - values.detach()).square()
        indices = distances.argmin(-1)
        assignments.append(indices)
        selected.append(values[indices])
    return latents, torch.stack(selected, -1), assignments, shape


def projected_reference(model, latents, selected, shape):
    # Reconstruction has an identity latent Jacobian and discrete values.
    straight_through = latents + (selected - latents).detach()
    rows = straight_through.flatten(-2)
    if model.has_projections:
        rows = nn.functional.linear(
            rows, model.project_out.weight, model.project_out.bias
        )
    return rows.reshape(shape).movedim(-1, 1)


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("num_codebooks", [1, 2])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("weights", [(0.3, 0.0), (0.0, 0.7), (0.3, 0.7)])
def test_learnable_scalar_values_receive_only_the_quantization_loss_gradient(
    projected, num_codebooks, image, weights
):
    torch.manual_seed(227)
    dim = 2 * num_codebooks + int(projected)
    commitment, quantization = weights
    model = LatentQuantize(
        levels=[3, 4],
        dim=dim,
        num_codebooks=num_codebooks,
        commitment_loss_weight=commitment,
        quantization_loss_weight=quantization,
    ).train()
    shape = (2, dim, 2, 3) if image else (2, dim, 4)
    inputs = (torch.randn(shape) * 0.2).requires_grad_()
    output, indices, loss = model(inputs)
    latents, selected, assignments, spatial = latent_reference(model, inputs)
    reference = projected_reference(model, latents, selected, spatial)
    expected_loss = (
        commitment * (latents - selected.detach()).square().mean()
        + quantization * (latents.detach() - selected).square().mean()
    )
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(loss, expected_loss)
    assert indices.shape == (
        (*inputs.shape[:1], *inputs.shape[2:], num_codebooks)
        if num_codebooks > 1
        else (*inputs.shape[:1], *inputs.shape[2:])
    )
    actual_input_grad = torch.autograd.grad(
        loss, inputs, retain_graph=True, allow_unused=True
    )[0]
    expected_input_grad = torch.autograd.grad(
        expected_loss, inputs, retain_graph=True, allow_unused=True
    )[0]
    actual_input_grad = (
        torch.zeros_like(inputs) if actual_input_grad is None else actual_input_grad
    )
    expected_input_grad = (
        torch.zeros_like(inputs) if expected_input_grad is None else expected_input_grad
    )
    torch.testing.assert_close(actual_input_grad, expected_input_grad)

    parameters = tuple(model.values_per_latent)
    actual_grads = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    expected_grads = []
    for axis, (values, assignment) in enumerate(zip(parameters, assignments)):
        gradient = torch.zeros_like(values)
        contributions = (
            2
            * quantization
            * (selected[..., axis].detach() - latents[..., axis].detach())
            / latents.numel()
        )
        gradient.scatter_add_(0, assignment.flatten(), contributions.flatten())
        expected_grads.append(gradient)
    for actual, expected in zip(actual_grads, expected_grads):
        actual = torch.zeros_like(expected) if actual is None else actual
        torch.testing.assert_close(actual, expected)

    actual_reconstruction_grad = torch.autograd.grad(
        output.sum(), inputs, retain_graph=True
    )[0]
    expected_reconstruction_grad = torch.autograd.grad(reference.sum(), inputs)[0]
    torch.testing.assert_close(actual_reconstruction_grad, expected_reconstruction_grad)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    loss.backward()
    optimizer.step()
    for parameter, original, gradient in zip(parameters, before, expected_grads):
        torch.testing.assert_close(parameter, original - 0.05 * gradient)


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("num_codebooks", [1, 2])
@pytest.mark.parametrize("training", [False, True])
def test_in_place_optimizer_updates_raw_codewords_without_consuming_encoder_graph(
    projected, num_codebooks, training
):
    torch.manual_seed(241)
    dim = 2 * num_codebooks + int(projected)
    model = LatentQuantize(
        levels=[3, 4],
        dim=dim,
        num_codebooks=num_codebooks,
        commitment_loss_weight=0.3,
        quantization_loss_weight=0.7,
        in_place_codebook_optimizer=partial(torch.optim.SGD, lr=0.05),
    ).train(training)
    inputs = (torch.randn(2, dim, 4) * 0.2).requires_grad_()
    latents, selected, assignments, spatial = latent_reference(model, inputs)
    before = [values.detach().clone() for values in model.values_per_latent]
    expected_values = []
    for axis, (values, assignment) in enumerate(zip(before, assignments)):
        gradient = torch.zeros_like(values)
        contributions = (
            1.4
            * (selected[..., axis].detach() - latents[..., axis].detach())
            / latents.numel()
        )
        gradient.scatter_add_(0, assignment.flatten(), contributions.flatten())
        expected_values.append(values - 0.05 * gradient if training else values)
    actual, _, loss = model(inputs)
    assert inputs.grad is None
    if model.has_projections:
        assert model.project_in.weight.grad is None
        assert model.project_out.weight.grad is None
    for values, expected in zip(model.values_per_latent, expected_values):
        torch.testing.assert_close(values, expected)
    latents, selected, _, spatial = latent_reference(model, inputs)
    torch.testing.assert_close(
        actual, projected_reference(model, latents, selected, spatial)
    )
    (actual.square().mean() + loss).backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("num_codebooks", [1, 2])
def test_fixed_codebook_values_still_train_encoder_with_commitment_objective(
    num_codebooks,
):
    model = LatentQuantize(
        levels=[3, 4],
        dim=2 * num_codebooks,
        num_codebooks=num_codebooks,
        optimize_values=False,
        commitment_loss_weight=0.3,
        quantization_loss_weight=0.7,
    ).train()
    inputs = (torch.randn(2, 2 * num_codebooks, 4) * 0.2).requires_grad_()
    output, _, loss = model(inputs)
    latents, selected, _, _ = latent_reference(model, inputs)
    expected = (
        0.3 * (latents - selected.detach()).square().mean()
        + 0.7 * (latents.detach() - selected).square().mean()
    )
    actual_grad = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    (output.square().mean() + loss).backward()
    assert torch.isfinite(inputs.grad).all()
