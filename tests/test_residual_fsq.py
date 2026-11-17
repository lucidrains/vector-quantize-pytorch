import pytest
import torch

from vector_quantize_pytorch import FSQ, ResidualFSQ


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_table_free_residual_decode(device):
    kwargs = dict(levels = [4, 5, 3], num_quantizers = 3, quantize_dropout = True)
    dense = ResidualFSQ(**kwargs).to(device)
    compact = ResidualFSQ(**kwargs, materialize_codebook = False).to(device)
    indices = torch.tensor([[[[0, 59], [21, -1]]]], device = device)
    torch.testing.assert_close(compact.get_codes_from_indices(indices), dense.get_codes_from_indices(indices))
    torch.testing.assert_close(compact.codebooks, dense.codebooks)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_large_grid(device):
    quantizer = FSQ([16] * 8, materialize_codebook = False).to(device)
    indices = torch.tensor([[0, 2 ** 24 + 1, 2 ** 31, 2 ** 32 - 1]], device = device)
    torch.testing.assert_close(quantizer.codes_to_indices(quantizer.indices_to_codes(indices)), indices)
    _, encoded = quantizer(torch.randn(2, 7, 8, device = device))
    assert encoded.dtype == torch.int64
