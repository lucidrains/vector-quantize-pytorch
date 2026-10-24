import pytest
import torch
from torch import nn
from vector_quantize_pytorch import ResidualLFQ


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("dropout", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_residual_lfq_decodes_physical_codewords_and_dropout_sentinels(
    projected, rotation, spherical, dropout, training
):
    torch.manual_seed(71)
    model = ResidualLFQ(
        dim=3 if projected else 2,
        codebook_size=4,
        num_quantizers=3,
        orthogonal_rotation=rotation,
        spherical=spherical,
        quantize_dropout=dropout,
    ).train(training)
    if rotation:
        with torch.no_grad():
            for layer in model.layers:
                layer.orthogonal_rot.copy_(torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
    inputs = torch.randn(
        2, 4, model.project_in.in_features if projected else 2, requires_grad=True
    )
    quantized, indices, losses, codes = model(
        inputs, return_all_codes=True, rand_quantize_dropout_fixed_seed=1
    )
    active = 1 if dropout and training else 3
    assert torch.equal(
        indices[..., active:], torch.full_like(indices[..., active:], -1)
    )

    # Four enumerated two-bit vectors, with each layer's physical scale and
    # normalization/rotation applied explicitly, independently of decode APIs.
    signs = inputs.new_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    tables = []
    for layer in model.layers:
        table = signs * layer.codebook_scale
        if spherical:
            table = table / table.norm(dim=-1, keepdim=True) * layer.codebook_scale
        if rotation:
            table = table @ layer.orthogonal_rot.t()
        tables.append(table)
    tables = torch.stack(tables)
    torch.testing.assert_close(model.codebooks, tables)
    expected_codes = []
    residual = inputs.detach()
    if projected:
        residual = nn.functional.linear(
            residual, model.project_in.weight, model.project_in.bias
        )
    bit_weights = torch.tensor([2, 1])
    for stage, layer in enumerate(model.layers):
        if stage >= active:
            expected_codes.append(torch.zeros_like(residual))
            continue
        rotated = residual @ layer.orthogonal_rot if rotation else residual
        expected_indices = ((rotated > 0).long() * bit_weights).sum(-1)
        assert torch.equal(indices[..., stage], expected_indices)
        selected = tables[stage][expected_indices]
        expected_codes.append(selected)
        residual = residual - selected
    expected_codes = torch.stack(expected_codes)
    expected = expected_codes.sum(0)
    if projected:
        expected = nn.functional.linear(
            expected, model.project_out.weight, model.project_out.bias
        )
    torch.testing.assert_close(codes, expected_codes)
    torch.testing.assert_close(quantized, expected)
    decoded = model.get_output_from_indices(indices)
    torch.testing.assert_close(decoded, expected)
    if dropout:
        torch.testing.assert_close(
            model.get_output_from_indices(indices[..., :active]), expected
        )
    if projected:
        actual_grad = torch.autograd.grad(
            decoded.square().sum(), model.project_out.weight
        )[0]
        expected_grad = torch.autograd.grad(
            expected.square().sum(), model.project_out.weight
        )[0]
        torch.testing.assert_close(actual_grad, expected_grad)
    if training:
        (quantized.square().mean() + losses.sum()).backward()
        assert torch.isfinite(inputs.grad).all()
    elif projected:
        quantized.square().mean().backward()
        assert torch.isfinite(model.project_out.weight.grad).all()
        assert inputs.grad is None
    else:
        assert not quantized.requires_grad
