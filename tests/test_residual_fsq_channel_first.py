import pytest
import torch

from vector_quantize_pytorch import GroupedResidualFSQ, ResidualFSQ


@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('spatial_shape', ((3,), (5,), (4, 3), (2, 4, 5)))
def test_channel_first_decode(dim, spatial_shape):
    torch.manual_seed(0)
    quantizer = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3, is_channel_first = True
    ).eval()
    x = torch.randn(2, dim, *spatial_shape)

    quantized, indices = quantizer(x)

    assert indices.shape == (2, 3, *spatial_shape)
    reconstructed = quantizer.get_output_from_indices(indices)
    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, quantized)


@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('spatial_shape', ((3,), (4, 3)))
def test_channel_first_codes_match_channel_last_reference(dim, spatial_shape):
    # Equal quantizer/spatial axis sizes must not silently exchange their values.
    torch.manual_seed(0)
    channel_first = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3, is_channel_first = True
    ).eval()
    channel_last = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3
    ).eval()
    channel_last.load_state_dict(channel_first.state_dict())
    x = torch.randn(2, dim, *spatial_shape)

    quantized, indices = channel_first(x)
    reference, reference_indices, reference_codes = channel_last(
        x.movedim(1, -1).flatten(1, -2), return_all_codes = True
    )

    torch.testing.assert_close(
        quantized.movedim(1, -1).flatten(1, -2), reference
    )
    torch.testing.assert_close(
        indices.movedim(1, -1).flatten(1, -2), reference_indices
    )
    codes = channel_first.get_codes_from_indices(indices)
    assert codes.shape == (3, 2, *spatial_shape, 2)
    torch.testing.assert_close(codes.flatten(2, -2), reference_codes)


@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('spatial_shape', ((5,), (4, 5), (2, 4, 5)))
def test_channel_first_return_all_codes(dim, spatial_shape):
    torch.manual_seed(0)
    quantizer = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3, is_channel_first = True
    ).eval()
    x = torch.randn(2, dim, *spatial_shape)

    quantized, indices, codes = quantizer(x, return_all_codes = True)

    assert codes.shape == (3, 2, *spatial_shape, 2)
    torch.testing.assert_close(codes, quantizer.get_codes_from_indices(indices))
    reconstructed = quantizer.project_out(codes.sum(0)).movedim(-1, 1)
    torch.testing.assert_close(reconstructed, quantized)


@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('spatial_shape', ((5,), (4, 5)))
def test_channel_first_dropout_and_coarse_indices(dim, spatial_shape):
    torch.manual_seed(0)
    quantizer = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3,
        is_channel_first = True, quantize_dropout = True
    )
    x = torch.randn(2, dim, *spatial_shape, requires_grad = True)

    # This seed keeps only the first quantizer.
    quantized, indices, codes = quantizer(
        x, return_all_codes = True, rand_quantize_dropout_fixed_seed = 1
    )

    assert (indices[:, 1:] == -1).all()
    assert torch.equal(codes[1:], torch.zeros_like(codes[1:]))
    torch.testing.assert_close(quantizer.get_output_from_indices(indices), quantized)
    torch.testing.assert_close(
        quantizer.get_output_from_indices(indices[:, :1]), quantized
    )
    torch.testing.assert_close(
        quantizer.get_codes_from_indices(indices[:, :1]), codes
    )

    quantized.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize('dim', (4, 8))
@pytest.mark.parametrize('spatial_shape', ((4, 5), (2, 4, 5)))
def test_grouped_channel_first_decode(dim, spatial_shape):
    torch.manual_seed(0)
    quantizer = GroupedResidualFSQ(
        dim = dim, groups = 2, levels = [3, 5], num_quantizers = 3,
        accept_image_fmap = True, is_channel_first = True
    ).eval()
    x = torch.randn(2, dim, *spatial_shape)

    quantized, indices, _ = quantizer(x, return_all_codes = True)

    assert indices.shape == (2, 2, 3, *spatial_shape)
    reconstructed = quantizer.get_output_from_indices(indices)
    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, quantized)


@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('quantize_dropout', (False, True))
def test_channel_last_decode_unchanged(dim, quantize_dropout):
    torch.manual_seed(0)
    quantizer = ResidualFSQ(
        dim = dim, levels = [3, 5], num_quantizers = 3,
        quantize_dropout = quantize_dropout
    )
    x = torch.randn(2, 5, dim)

    quantized, indices, codes = quantizer(
        x, return_all_codes = True, rand_quantize_dropout_fixed_seed = 1
    )

    assert indices.shape == (2, 5, 3)
    assert codes.shape == (3, 2, 5, 2)
    torch.testing.assert_close(quantizer.get_output_from_indices(indices), quantized)
    torch.testing.assert_close(quantizer.project_out(codes.sum(0)), quantized)
