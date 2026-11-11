import pytest
import torch

from vector_quantize_pytorch import FSQ


@pytest.mark.parametrize("preserve_symmetry", [False, True])
@pytest.mark.parametrize("orthogonal_rotation", [False, True])
def test_fsq_code_index_roundtrip_with_orthogonal_rotation(
    preserve_symmetry, orthogonal_rotation
):
    torch.manual_seed(0)
    quantizer = FSQ(
        levels=[5, 5],
        preserve_symmetry=preserve_symmetry,
        orthogonal_rotation=orthogonal_rotation,
    )
    indices = torch.arange(25).reshape(1, -1)
    codes = quantizer.indices_to_codes(indices)
    torch.testing.assert_close(
        quantizer.codes_to_indices(codes), indices.to(torch.int32)
    )


@pytest.mark.parametrize("preserve_symmetry", [False, True])
def test_rotated_fsq_forward_returns_indices_for_physical_codes(preserve_symmetry):
    torch.manual_seed(0)
    quantizer = FSQ(
        levels=[5, 5], preserve_symmetry=preserve_symmetry, orthogonal_rotation=True
    )
    quantized, indices = quantizer(torch.randn(2, 16, 2))
    torch.testing.assert_close(quantizer.indices_to_codes(indices), quantized)
    torch.testing.assert_close(quantizer.codes_to_indices(quantized), indices)
