import pytest
import torch
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import orthogonal_loss_fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "codes",
    [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[5e-7, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ],
)
def test_orthogonal_loss_matches_squared_gram_error_with_zero_codes(dtype, codes):
    values = torch.tensor([codes], dtype=dtype, requires_grad=True)
    actual = orthogonal_loss_fn(values)
    normalized = F.normalize(values, dim=-1, eps=1e-6)
    gram = normalized @ normalized.transpose(-1, -2)
    expected = (gram - torch.eye(gram.shape[-1], dtype=dtype)).square().mean()
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual, values, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_native_vector_quantize_zero_codebook_has_positive_orthogonal_error():
    quantizer = VectorQuantize(
        dim=2,
        codebook_size=4,
        learnable_codebook=True,
        ema_update=False,
        commitment_weight=0,
        orthogonal_reg_weight=1,
        orthogonal_reg_max_codes=None,
        threshold_ema_dead_code=0,
        rotation_trick=False,
    )
    with torch.no_grad():
        quantizer.codebook.zero_()
    _, _, loss = quantizer(torch.zeros(2, 3, 2))
    torch.testing.assert_close(loss, torch.tensor(1 / quantizer.codebook_size))
