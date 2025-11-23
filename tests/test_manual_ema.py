import torch
from vector_quantize_pytorch import VectorQuantize

def test_manual_ema_update():

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
    quantize1, indices1, _ = vq1(x, mask = mask)

    vq2.train()
    quantize2, indices2, _ = vq2(x, mask = mask, ema_update = False)

    assert torch.allclose(quantize1, quantize2)
    assert torch.equal(indices1, indices2)

    assert not torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)

    vq2.update_ema_indices(x, indices2, mask = mask)

    assert torch.allclose(vq1._codebook.cluster_size, vq2._codebook.cluster_size)
    assert torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)
    assert torch.allclose(vq1.codebook, vq2.codebook)
