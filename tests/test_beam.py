import pytest
param = pytest.mark.parametrize

import torch
from vector_quantize_pytorch import VectorQuantize

def test_topk_and_manual_ema_update():

    vq1 = VectorQuantize(
        dim = 256,
        codebook_size = 512
    )

    vq2 = VectorQuantize(
        dim = 256,
        codebook_size = 512
    )

    vq2.load_state_dict(vq1.state_dict())

    x = torch.randn(1, 1024, 256)
    mask = torch.randint(0, 2, (1, 1024)).bool()

    vq1.train()
    quantize1, indices1, commit_loss1 = vq1(x, mask = mask)

    vq2.train()
    quantize2, indices2, commit_losses = vq2(x, mask = mask, topk = 1, ema_update = False)

    assert quantize2.shape == (1, 1024, 1, 256)
    assert indices2.shape == (1, 1024, 1)
    assert commit_losses.shape == (1, 1024, 1)

    top_quantize2 = quantize2[..., 0, :]
    top_indices2 = indices2[..., 0]

    assert torch.allclose(commit_loss1, commit_losses.sum() / mask.sum())
    assert torch.equal(indices1, top_indices2)
    assert torch.allclose(quantize1, top_quantize2)

    assert not torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)

    vq2.update_ema_indices(x, top_indices2, mask = mask)

    assert torch.allclose(vq1._codebook.cluster_size, vq2._codebook.cluster_size)
    assert torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)
    assert torch.allclose(vq1.codebook, vq2.codebook)

@param('training', (False, True))
@param('topk', (1, 2))
@param('transform_codebook', (False, True))
@param(
    'heads,separate_codebook_per_head',
    (
        (1, False),
        (1, True),
        (2, False),
        (2, True)
    )
)
def test_topk_quantization_in_train_and_eval(
    training,
    topk,
    transform_codebook,
    heads,
    separate_codebook_per_head
):
    vq = VectorQuantize(
        dim = 8,
        codebook_dim = 3,
        heads = heads,
        separate_codebook_per_head = separate_codebook_per_head,
        codebook_size = 8,
    )

    vq.train(training)

    x = torch.randn(2, 3, 8, requires_grad = training)

    codebook_transform_fn = None

    if transform_codebook:
        transform_batch = x.shape[0]

        if heads > 1 and not separate_codebook_per_head:
            transform_batch *= heads

        def codebook_transform_fn(codebook):
            return codebook[:, None, None].expand(
                -1, transform_batch, x.shape[1], -1, -1
            )

    forward_kwargs = {
        'freeze_codebook': True,
        'ema_update': False,
        'codebook_transform_fn': codebook_transform_fn
    }

    baseline_quantized, baseline_indices, _ = vq(x, **forward_kwargs)
    quantized, indices, commit_loss = vq(x, topk = topk, **forward_kwargs)

    expected_indices_shape = (2, 3, topk)
    expected_baseline_indices_shape = (2, 3)
    top_indices = indices[..., 0]

    if heads > 1:
        expected_indices_shape = (*expected_indices_shape, heads)
        expected_baseline_indices_shape = (*expected_baseline_indices_shape, heads)
        top_indices = indices[..., 0, :]

    assert baseline_quantized.shape == (2, 3, 8)
    assert baseline_indices.shape == expected_baseline_indices_shape
    assert quantized.shape == (2, 3, topk, 8)
    assert indices.shape == expected_indices_shape
    assert quantized.dtype == x.dtype
    assert indices.dtype == torch.long
    assert quantized.device == x.device
    assert indices.device == x.device
    assert torch.allclose(quantized[..., 0, :], baseline_quantized)
    assert torch.equal(top_indices, baseline_indices)

    if training:
        assert commit_loss.shape == (2, 3, topk)

        (quantized.square().mean() + commit_loss.mean()).backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

@param('layout', ('channel_first', 'image', 'video', 'single_token'))
@param('separate_codebook_per_head', (False, True))
def test_multiheaded_topk_output_layouts(layout, separate_codebook_per_head):
    vq_kwargs = {
        'dim': 8,
        'codebook_dim': 3,
        'codebook_size': 8,
        'heads': 2,
        'separate_codebook_per_head': separate_codebook_per_head
    }

    if layout == 'channel_first':
        vq_kwargs.update(channel_last = False)
        x = torch.randn(2, 8, 3)
        expected_quantized_shape = (2, 8, 3, 2)
        expected_indices_shape = (2, 3, 2, 2)
    elif layout == 'image':
        vq_kwargs.update(accept_image_fmap = True)
        x = torch.randn(2, 8, 2, 3)
        expected_quantized_shape = (2, 8, 2, 3, 2)
        expected_indices_shape = (2, 2, 3, 2, 2)
    elif layout == 'video':
        vq_kwargs.update(accept_3d_fmap = True)
        x = torch.randn(2, 8, 2, 2, 3)
        expected_quantized_shape = (2, 8, 2, 2, 3, 2)
        expected_indices_shape = (2, 2, 2, 3, 2, 2)
    else:
        x = torch.randn(2, 8)
        expected_quantized_shape = (2, 2, 8)
        expected_indices_shape = (2, 2, 2)

    vq = VectorQuantize(**vq_kwargs).eval()

    baseline_quantized, baseline_indices, _ = vq(x)
    quantized, indices, _ = vq(x, topk = 2)

    top_quantized = quantized[:, 0] if layout == 'single_token' else quantized[..., 0]
    top_indices = indices.select(-2, 0)

    assert baseline_quantized.shape == x.shape
    assert baseline_indices.shape == expected_indices_shape[:-2] + (2,)
    assert quantized.shape == expected_quantized_shape
    assert indices.shape == expected_indices_shape
    assert quantized.dtype == x.dtype
    assert indices.dtype == torch.long
    assert quantized.device == x.device
    assert indices.device == x.device
    assert torch.allclose(top_quantized, baseline_quantized)
    assert torch.equal(top_indices, baseline_indices)

    decoded = vq.get_output_from_indices(baseline_indices)
    assert decoded.shape == baseline_quantized.shape
    assert torch.allclose(decoded, baseline_quantized)

@param('codebook_dim', (256, 128))
def test_beam_search(
    codebook_dim
):
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        codebook_dim = codebook_dim,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
        quantize_dropout = True,
        beam_size = 2,
        eval_beam_size = 3
    )

    x = torch.randn(1, 1024, 256).requires_grad_()

    for _ in range(5):
        quantized, indices, commit_loss = residual_vq(x)

    assert quantized.shape == (1, 1024, 256)
    assert indices.shape == (1, 1024, 8)
    assert commit_loss.shape == (8,)
