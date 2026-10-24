import pytest
import torch
from torch import nn

from vector_quantize_pytorch import LFQ, EvoLFQ


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("num_codebooks", [1, 2])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("return_signs", [False, True])
def test_encode_returns_actual_code_bits_before_projection_and_rotation(
    projected, num_codebooks, vector, rotation, return_signs
):
    torch.manual_seed(41)
    code_dim = num_codebooks * 3
    dim = code_dim + int(projected)
    lfq = LFQ(
        dim=dim,
        codebook_size=8,
        num_codebooks=num_codebooks,
        orthogonal_rotation=rotation,
    )
    if projected:
        with torch.no_grad():
            lfq.project_in.weight.zero_()
            lfq.project_in.weight[:, :code_dim].copy_(torch.eye(code_dim))
            lfq.project_in.bias.zero_()
            lfq.project_out.weight.copy_(
                torch.linspace(-0.2, 0.3, dim * code_dim).reshape(dim, code_dim)
            )
            lfq.project_out.bias.copy_(torch.linspace(0.5, 1.0, dim))
    model = EvoLFQ(nn.Identity(), nn.Identity(), lfq=lfq).eval()
    spatial = () if vector else (3,)
    signs = (
        torch.arange(2 * (1 if vector else 3) * code_dim)
        .reshape(2, *spatial, num_codebooks, 3)
        .remainder(2)
        .float()
        * 2
        - 1
    )
    projected_inputs = signs @ lfq.orthogonal_rot.t() if rotation else signs
    inputs = projected_inputs.flatten(-2)
    if projected:
        inputs = torch.cat((inputs, torch.full((*inputs.shape[:-1], 1), 0.25)), -1)
    expected_bits = signs.flatten(-2)
    if not return_signs:
        expected_bits = (expected_bits > 0).float()
    actual = model.encode(inputs, return_signs=return_signs)
    torch.testing.assert_close(actual, expected_bits)

    # Reconstruct directly in latent space, then apply the known LFQ output
    # projection. This oracle does not call native index/bit decode helpers.
    expected_codes = signs @ lfq.orthogonal_rot.t() if rotation else signs
    expected_codes = expected_codes.flatten(-2)
    if projected:
        expected_codes = nn.functional.linear(
            expected_codes, lfq.project_out.weight, lfq.project_out.bias
        )
    torch.testing.assert_close(model.decode_bits(actual), expected_codes)
    torch.testing.assert_close(model(inputs).reconstructed, expected_codes)
    population = model.init_population_from_data(
        inputs, pop_size=4, mutation_rate=0.0, is_sign=return_signs
    )
    torch.testing.assert_close(
        population, expected_bits.repeat(2, *([1] * (expected_bits.ndim - 1)))
    )
