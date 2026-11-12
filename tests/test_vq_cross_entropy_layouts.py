import pytest
import torch

from vector_quantize_pytorch import VectorQuantize


@pytest.mark.parametrize('layout', ('single', 'sequence', 'image', 'volume'))
@pytest.mark.parametrize('heads, separate', ((1, False), (2, False), (2, True)))
@pytest.mark.parametrize('supervised', (False, True))
def test_cross_entropy_uses_the_public_index_layout(layout, heads, separate, supervised):
    torch.manual_seed(17)
    quantizer = VectorQuantize(
        dim = 4, codebook_dim = 4 // heads, codebook_size = 5,
        heads = heads, separate_codebook_per_head = separate,
        freeze_codebook = True, rotation_trick = False,
        commitment_use_cross_entropy_loss = True, commitment_weight = .7,
        accept_image_fmap = layout == 'image', accept_3d_fmap = layout == 'volume'
    ).eval()
    sizes = dict(single = (2, 4), sequence = (2, 3, 4), image = (2, 4, 2, 3), volume = (2, 4, 2, 2, 3))
    values = torch.randn(sizes[layout], requires_grad = True)
    _, targets, _ = quantizer(values.detach())
    quantizer.train()

    result = quantizer(values, indices = targets) if supervised else quantizer(values)
    loss = result[-1]

    tokens = values.movedim(1, -1) if layout in ('image', 'volume') else values
    tokens = tokens.reshape(2, -1, heads, 4 // heads)
    flat_targets = targets.reshape(2, -1, heads)
    per_head_losses = []
    for head in range(heads):
        codes = quantizer._codebook.embed[head if separate else 0].detach()
        logits = -(tokens[:, :, head, None] - codes).norm(dim = -1)
        nll = -logits.log_softmax(-1).gather(-1, flat_targets[:, :, head, None])
        per_head_losses.append(nll.mean())
    expected = torch.stack(per_head_losses).mean() * (1. if supervised else .7)
    torch.testing.assert_close(loss.squeeze(), expected)
    actual_gradient, = torch.autograd.grad(loss.sum(), values, retain_graph = True)
    expected_gradient, = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual_gradient, expected_gradient)
