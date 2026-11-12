import pytest
import torch

from vector_quantize_pytorch import FSP


@pytest.mark.parametrize('levels', ([8, 5, 5, 5], [8] * 6))
@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_fsp_encodes_level_indices_exactly(levels, dtype):
    fsp = FSP(levels = levels, vector_norm = 'none')
    digits = torch.tensor([[level - 1 for level in levels]], dtype = dtype)

    indices = fsp.level_indices_to_indices(digits)

    assert indices.dtype == torch.int32
    assert indices.item() == fsp.codebook_size - 1
    torch.testing.assert_close(
        fsp.indices_to_level_indices(indices), digits.to(torch.int32)
    )


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
@pytest.mark.parametrize('channel_first', (False, True))
def test_fsp_half_precision_indices_reconstruct_quantized_codes(dtype, channel_first):
    fsp = FSP(levels = [8] * 6, channel_first = channel_first, vector_norm = 'none').eval()
    x = torch.tensor(
        [[[-3., 3., -3., 3., -3., 3.], [3., 3., 3., 3., 3., 3.]]],
        dtype = dtype
    )
    if channel_first:
        x = x.transpose(1, 2)

    quantized, indices, _, info = fsp(x)
    digits = info['level_indices'].to(torch.int64)
    weights = torch.tensor([8 ** power for power in range(6)], dtype = torch.int64)
    expected_indices = (digits * weights).sum(dim = -1)

    torch.testing.assert_close(indices.to(torch.int64), expected_indices)
    assert ((indices >= 0) & (indices < fsp.codebook_size)).all()
    torch.testing.assert_close(
        fsp.indices_to_codes(indices), quantized.float(), atol = 2e-2, rtol = 2e-2
    )
