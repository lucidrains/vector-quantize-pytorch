import pytest
import torch
from torch import nn

from vector_quantize_pytorch import SimVQ, VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('scale', (0., 1e-10, 1e-7, 1.))
def test_rotation_keeps_selected_code_values(dtype, scale):
    source = torch.tensor([[[3., 4.], [-4., 3.]]], dtype = dtype)
    source = (source * scale).requires_grad_()
    target = torch.tensor([[[-2., 0.], [0., 2.]]], dtype = dtype)

    actual = rotate_to(source, target)

    torch.testing.assert_close(actual, target)
    actual.sum().backward()
    assert torch.isfinite(source.grad).all()


def test_rotation_preserves_the_analytical_input_gradient():
    source = torch.tensor([[3., 4.]], dtype = torch.float64, requires_grad = True)
    target = torch.tensor([[-2., 0.]], dtype = torch.float64, requires_grad = True)
    upstream_gradient = torch.tensor([[1., 2.]], dtype = torch.float64)

    result = rotate_to(source, target)
    (result * upstream_gradient).sum().backward()

    # Rotating (3, 4) to (-2, 0) gives J = .4 * [[-.6, -.8], [.8, -.6]].
    torch.testing.assert_close(source.grad, torch.tensor([[.4, -.8]], dtype = torch.float64))
    assert target.grad is None


@pytest.mark.parametrize('kind', ('vq', 'sim_vq'))
@pytest.mark.parametrize('scale', (0., 1e-9))
def test_quantizer_output_matches_selected_codes_for_small_encoder_outputs(kind, scale):
    codebook = torch.tensor([[1., 0.], [0., 2.], [-3., 1.]])
    if kind == 'vq':
        quantizer = VectorQuantize(
            dim = 2,
            codebook_size = 3,
            rotation_trick = True,
            commitment_weight = 0.,
            freeze_codebook = True
        )
        quantizer.codebook = codebook
    else:
        quantizer = SimVQ(dim = 2, codebook_size = 3, codebook_transform = nn.Identity())
        quantizer.frozen_codebook.copy_(codebook)

    source = torch.full((2, 3, 2), scale, requires_grad = True)
    quantized, indices, _ = quantizer(source)

    torch.testing.assert_close(indices, torch.zeros((2, 3), dtype = indices.dtype))
    torch.testing.assert_close(quantized, codebook[indices], rtol = 0., atol = 0.)
    quantized.sum().backward()
    assert torch.isfinite(source.grad).all()
