import pytest
import torch

from vector_quantize_pytorch import ResidualSimVQ


@pytest.mark.parametrize("spatial_shape", [(), (4,), (2, 3), (2, 2, 3)])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("multiple_of", [1, 2])
def test_dropout_preserves_all_token_axes_and_active_codebook_values(
    spatial_shape, channel_first, multiple_of
):
    torch.manual_seed(23)
    model = ResidualSimVQ(
        dim=3,
        codebook_size=7,
        num_quantizers=3,
        channel_first=channel_first,
        rotation_trick=False,
        quantize_dropout=True,
        quantize_dropout_multiple_of=multiple_of,
    ).train()
    input_shape = (2, 3, *spatial_shape) if channel_first else (2, *spatial_shape, 3)
    inputs = torch.randn(input_shape, requires_grad=True)
    quantized, indices, losses, codes = model(
        inputs, return_all_codes=True, rand_quantize_dropout_fixed_seed=1
    )
    active = multiple_of
    assert indices.shape == (2, *spatial_shape, 3)
    null_indices = torch.full_like(indices[..., active:], -1)
    assert torch.equal(indices[..., active:], null_indices)
    torch.testing.assert_close(losses[active:], torch.zeros_like(losses[active:]))

    # Independent nearest-neighbor oracle uses squared distances, then sums the
    # selected transformed table rows. It never calls native decode helpers.
    residual = inputs.detach().movedim(1, -1) if channel_first else inputs.detach()
    expected_codes = []
    tables = model.codebooks.detach()
    for layer in range(3):
        if layer >= active:
            expected_codes.append(torch.zeros_like(residual))
            continue
        distances = (residual[..., None, :] - tables[layer]).square().sum(-1)
        expected_indices = distances.argmin(-1)
        assert torch.equal(indices[..., layer], expected_indices)
        selected = tables[layer][expected_indices]
        expected_codes.append(selected)
        residual = residual - selected
    expected_codes = torch.stack(expected_codes)
    expected = expected_codes.sum(0)
    if channel_first:
        expected_codes = expected_codes.movedim(-1, 2)
        expected = expected.movedim(-1, 1)
    torch.testing.assert_close(codes, expected_codes)
    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(model.get_output_from_indices(indices), expected)

    # Each active straight-through quantizer contributes one identity Jacobian.
    gradient = torch.autograd.grad(quantized.sum(), inputs, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.full_like(inputs, active))
    (quantized.square().mean() + losses.sum()).backward()
    assert torch.isfinite(inputs.grad).all()
    for layer in model.layers[:active]:
        assert torch.isfinite(layer.code_transform.weight.grad).all()
    for layer in model.layers[active:]:
        assert layer.code_transform.weight.grad is None


@pytest.mark.parametrize("channel_first", [False, True])
def test_dropout_disabled_image_forward_matches_codebook_reconstruction(channel_first):
    model = ResidualSimVQ(
        dim=3,
        codebook_size=7,
        num_quantizers=3,
        channel_first=channel_first,
        rotation_trick=False,
    ).eval()
    shape = (2, 3, 2, 4) if channel_first else (2, 2, 4, 3)
    quantized, indices, _ = model(torch.randn(shape))
    assert indices.shape == (2, 2, 4, 3)
    torch.testing.assert_close(model.get_output_from_indices(indices), quantized)
