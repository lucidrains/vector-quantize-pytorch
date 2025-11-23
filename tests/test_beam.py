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

def test_beam_search():
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
        quantize_dropout = True,
        beam_size = 2,
        eval_beam_size = 3
    )

    x = torch.randn(1, 1024, 256)

    for _ in range(5):
        quantized, indices, commit_loss = residual_vq(x)

    assert quantized.shape == (1, 1024, 256)
    assert indices.shape == (1, 1024, 8)
    assert commit_loss.shape == (8,)
