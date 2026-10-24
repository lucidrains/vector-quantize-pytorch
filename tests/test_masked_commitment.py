import pytest
import torch
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize(
    "heads,codebook_dim,separate_codebook_per_head",
    (
        (1, 6, False),
        (1, 3, False),
        (2, 3, False),
        (2, 3, True),
        (2, 2, False),
        (2, 2, True),
    ),
)
@pytest.mark.parametrize("use_cosine_sim", (False, True))
@pytest.mark.parametrize("channel_last", (False, True))
@pytest.mark.parametrize("with_padding", (False, True))
def test_masked_commitment_in_codebook_space(
    *,
    heads,
    codebook_dim,
    separate_codebook_per_head,
    use_cosine_sim,
    channel_last,
    with_padding,
):
    torch.manual_seed(0)
    vq = VectorQuantize(
        dim=6,
        codebook_dim=codebook_dim,
        codebook_size=8,
        heads=heads,
        separate_codebook_per_head=separate_codebook_per_head,
        use_cosine_sim=use_cosine_sim,
        channel_last=channel_last,
        rotation_trick=False,
        commitment_weight=0.7,
    ).train()

    x = (3 * torch.randn(2, 4, 6)).requires_grad_()
    mask = torch.ones(2, 4, dtype=torch.bool)
    if with_padding:
        mask[0, 3] = False
        mask[1, 2:] = False

    valid = x[mask]
    packed_input = valid.unsqueeze(0)
    quantized, indices, loss, breakdown = vq(
        x if channel_last else x.transpose(1, 2),
        mask=mask,
        freeze_codebook=True,
        return_loss_breakdown=True,
    )
    packed_quantized, packed_indices, packed_loss = vq(
        packed_input if channel_last else packed_input.transpose(1, 2),
        freeze_codebook=True,
    )

    # Independent squared-distance lookup in the projected, per-head code space.
    projected = vq.project_in(valid).reshape(-1, heads, codebook_dim)
    if use_cosine_sim:
        projected = F.normalize(projected, dim=-1, eps=1e-6)

    expected_indices = []
    expected_codes = []
    for head in range(heads):
        codebook = vq._codebook.embed[head if separate_codebook_per_head else 0]
        codebook = codebook.detach()
        distances = (projected[:, head, None] - codebook[None]).square().sum(-1)
        selected = distances.argmin(-1)
        expected_indices.append(selected)
        expected_codes.append(codebook[selected])

    expected_indices = torch.stack(expected_indices, dim=-1)
    if heads == 1:
        expected_indices = expected_indices.squeeze(-1)
    expected_codes = torch.stack(expected_codes, dim=1)
    expected_commitment = F.mse_loss(expected_codes, projected)
    expected_quantized = vq.project_out(
        expected_codes.reshape(-1, heads * codebook_dim)
    )

    if not channel_last:
        quantized = quantized.transpose(1, 2)
        packed_quantized = packed_quantized.transpose(1, 2)

    torch.testing.assert_close(indices[mask], expected_indices)
    torch.testing.assert_close(quantized[mask], expected_quantized)
    torch.testing.assert_close(breakdown.commitment, expected_commitment)
    torch.testing.assert_close(loss, expected_commitment * vq.commitment_weight)
    torch.testing.assert_close(loss, packed_loss)
    torch.testing.assert_close(indices[mask], packed_indices.squeeze(0))
    torch.testing.assert_close(quantized[mask], packed_quantized.squeeze(0))
    torch.testing.assert_close(quantized[~mask], torch.zeros_like(quantized[~mask]))
    torch.testing.assert_close(indices[~mask], torch.full_like(indices[~mask], -1))

    targets = (x, *vq.project_in.parameters())
    actual_gradients = torch.autograd.grad(loss, targets)
    expected_gradients = torch.autograd.grad(
        expected_commitment * vq.commitment_weight, targets
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual_gradients[0][~mask], torch.zeros_like(actual_gradients[0][~mask])
    )
