import pytest
import torch
import torch.nn.functional as F

from vector_quantize_pytorch import RandomProjectionQuantizer


def independent_cosine_scores(quantizer, x, norm):
    if norm:
        x = F.layer_norm(x, (x.shape[-1],))

    scores = []
    for projection, codebook in zip(quantizer.rand_projs, quantizer.vq.codebook):
        projected = F.normalize(x @ projection, dim = -1)
        normalized_codebook = F.normalize(codebook, dim = -1)
        scores.append(projected @ normalized_codebook.transpose(-1, -2))

    return torch.stack(scores, dim = -2)


@pytest.mark.parametrize('num_codebooks', (1, 2, 4))
@pytest.mark.parametrize('norm', (True, False))
def test_random_projection_codebook_dimensions(num_codebooks, norm):
    quantizer = RandomProjectionQuantizer(
        dim = 6,
        codebook_dim = 3,
        codebook_size = 7,
        num_codebooks = num_codebooks,
        norm = norm
    )

    assert quantizer.vq.codebook.shape == (num_codebooks, 7, 3)
    assert not quantizer.vq.has_projections
    assert sum(parameter.numel() for parameter in quantizer.parameters()) == 0


@pytest.mark.parametrize('num_codebooks', (1, 2, 4))
@pytest.mark.parametrize('norm', (True, False))
def test_random_projection_indices_match_independent_codebooks(num_codebooks, norm):
    torch.manual_seed(42)
    quantizer = RandomProjectionQuantizer(
        dim = 6,
        codebook_dim = 3,
        codebook_size = 7,
        num_codebooks = num_codebooks,
        norm = norm
    )
    x = torch.randn(2, 11, 6)

    expected = independent_cosine_scores(quantizer, x, norm).argmax(dim = -1)
    if num_codebooks == 1:
        expected = expected.squeeze(-1)

    torch.testing.assert_close(quantizer(x), expected)


@pytest.mark.parametrize('num_codebooks', (1, 2, 4))
@pytest.mark.parametrize('norm', (True, False))
def test_random_projection_target_loss_matches_independent_codebooks(
    num_codebooks,
    norm
):
    torch.manual_seed(42)
    quantizer = RandomProjectionQuantizer(
        dim = 6,
        codebook_dim = 3,
        codebook_size = 7,
        num_codebooks = num_codebooks,
        norm = norm
    )
    x = torch.randn(2, 11, 6, requires_grad = True)
    scores = independent_cosine_scores(quantizer, x, norm)
    targets = scores.argmax(dim = -1)
    targets[0, 0] = -1
    expected_loss = F.cross_entropy(
        scores.reshape(-1, 7),
        targets.reshape(-1),
        ignore_index = -1
    )
    if num_codebooks == 1:
        targets = targets.squeeze(-1)

    codebook_before = quantizer.vq.codebook.clone()
    projections_before = quantizer.rand_projs.clone()
    loss = quantizer(x, indices = targets)

    torch.testing.assert_close(loss, expected_loss)
    expected_grad, = torch.autograd.grad(expected_loss, x)
    actual_grad, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(quantizer.vq.codebook, codebook_before)
    torch.testing.assert_close(quantizer.rand_projs, projections_before)


def test_random_projection_heads_are_independent():
    torch.manual_seed(42)
    quantizer = RandomProjectionQuantizer(
        dim = 6,
        codebook_dim = 3,
        codebook_size = 7,
        num_codebooks = 2,
        norm = False
    )
    # Project disjoint input features into the two codebooks.
    quantizer.rand_projs.copy_(torch.eye(6).reshape(2, 3, 6).transpose(1, 2))
    x = torch.randn(1, 64, 6)
    x[..., 3:] = torch.tensor([0., 1., 0.])

    indices = quantizer(x)

    assert indices[..., 0].unique().numel() > 1
    torch.testing.assert_close(indices[..., 1], indices[:, :1, 1].expand(1, 64))
