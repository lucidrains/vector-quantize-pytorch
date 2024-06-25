import torch
import pytest

def exists(v):
    return v is not None

@pytest.mark.parametrize('use_cosine_sim', (True, False))
def test_vq(
    use_cosine_sim
):
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,                # codebook size
        decay = 0.8,                        # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = 1.,             # the weight on the commitment loss
        use_cosine_sim = use_cosine_sim
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = vq(x)

def test_vq_eval():
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,     # codebook size
        decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = 1.   # the weight on the commitment loss
    )

    x = torch.randn(1, 1024, 256)

    vq.eval()
    quantized, indices, commit_loss = vq(x)
    assert torch.allclose(quantized, vq.get_output_from_indices(indices))

def test_residual_vq():
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
    )

    x = torch.randn(1, 1024, 256)

    quantized, indices, commit_loss = residual_vq(x)
    quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)

def test_residual_vq2():
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        num_quantizers = 8,
        codebook_size = 1024,
        stochastic_sample_codes = True,
        sample_codebook_temp = 0.1,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook = True              # whether to share the codebooks for all quantizers or not
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = residual_vq(x)


def test_grouped_residual_vq():
    import torch
    from vector_quantize_pytorch import GroupedResidualVQ

    residual_vq = GroupedResidualVQ(
        dim = 256,
        num_quantizers = 8,      # specify number of quantizers
        groups = 2,
        codebook_size = 1024,    # codebook size
    )

    x = torch.randn(1, 1024, 256)

    quantized, indices, commit_loss = residual_vq(x)

def test_residual_vq3():
    import torch
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        codebook_size = 256,
        num_quantizers = 4,
        kmeans_init = True,   # set to True
        kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = residual_vq(x)

def test_vq_lower_codebook():
    import torch
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 256,
        codebook_dim = 16      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = vq(x)

def test_vq_cosine_sim():
    import torch
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 256,
        use_cosine_sim = True   # set this to True
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = vq(x)

def test_vq_expire_code():
    import torch
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,
        threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = vq(x)

def test_vq_multiheaded():
    import torch
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim = 256,
        codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
        heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
        separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        codebook_size = 8196,
        accept_image_fmap = True
    )

    img_fmap = torch.randn(1, 256, 32, 32)
    quantized, indices, loss = vq(img_fmap)

def test_rq():
    import torch
    from vector_quantize_pytorch import RandomProjectionQuantizer

    quantizer = RandomProjectionQuantizer(
        dim = 512,               # input dimensions
        num_codebooks = 16,      # in USM, they used up to 16 for 5% gain
        codebook_dim = 256,      # codebook dimension
        codebook_size = 1024     # codebook size
    )

    x = torch.randn(1, 1024, 512)
    indices = quantizer(x)

def test_fsq():
    import torch
    from vector_quantize_pytorch import FSQ

    levels = [8,5,5,5] # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    assert torch.all(xhat == quantizer.indices_to_codes(indices))

def test_fsq_without_indices():
    import torch
    from vector_quantize_pytorch import FSQ

    levels = [8,5,5,5] # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels, return_indices = False)

    x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    assert not exists(indices)

def test_rfsq():
    import torch
    from vector_quantize_pytorch import ResidualFSQ

    residual_fsq = ResidualFSQ(
        dim = 256,
        levels = [8, 5, 5, 3],
        num_quantizers = 8
    )

    x = torch.randn(1, 1024, 256)

    residual_fsq.eval()

    quantized, indices = residual_fsq(x)

    quantized_out = residual_fsq.get_output_from_indices(indices)

    assert torch.all(quantized == quantized_out)

def test_lfq():
    import torch
    from vector_quantize_pytorch import LFQ

    # you can specify either dim or codebook_size
    # if both specified, will be validated against each other

    quantizer = LFQ(
        codebook_size = 65536,      # codebook size, must be a power of 2
        dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
        diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    )

    image_feats = torch.randn(1, 16, 32, 32)

    quantized, indices, entropy_aux_loss = quantizer(image_feats, inv_temperature=100.)  # you may want to experiment with temperature

    assert (quantized == quantizer.indices_to_codes(indices)).all()


def test_lfq_video():
    import torch
    from vector_quantize_pytorch import LFQ

    quantizer = LFQ(
        codebook_size = 65536,
        dim = 16,
        entropy_loss_weight = 0.1,
        diversity_gamma = 1.
    )

    seq = torch.randn(1, 32, 16)
    quantized, *_ = quantizer(seq)

    assert seq.shape == quantized.shape

    video_feats = torch.randn(1, 16, 10, 32, 32)
    quantized, *_ = quantizer(video_feats)

    assert video_feats.shape == quantized.shape


def test_lfq2():
    import torch
    from vector_quantize_pytorch import LFQ

    quantizer = LFQ(
        codebook_size = 4096,
        dim = 16,
        num_codebooks = 4  # 4 codebooks, total codebook dimension is log2(4096) * 4
    )

    image_feats = torch.randn(1, 16, 32, 32)

    quantized, indices, entropy_aux_loss = quantizer(image_feats)

    assert image_feats.shape == quantized.shape
    assert (quantized == quantizer.indices_to_codes(indices)).all()

def test_rflq():
    import torch
    from vector_quantize_pytorch import ResidualLFQ

    residual_lfq = ResidualLFQ(
        dim = 256,
        codebook_size = 256,
        num_quantizers = 8
    )

    x = torch.randn(1, 1024, 256)

    residual_lfq.eval()

    quantized, indices, commit_loss = residual_lfq(x)

    quantized_out = residual_lfq.get_output_from_indices(indices)

    assert torch.all(quantized == quantized_out)

def test_latent_q():
    import torch
    from vector_quantize_pytorch import LatentQuantize

    # you can specify either dim or codebook_size
    # if both specified, will be validated against each other

    quantizer = LatentQuantize(
        levels = [5, 5, 8],      # number of levels per codebook dimension
        dim = 16,                   # input dim
        commitment_loss_weight=0.1,  
        quantization_loss_weight=0.1,
    )

    image_feats = torch.randn(1, 16, 32, 32)

    quantized, indices, loss = quantizer(image_feats)

    assert image_feats.shape == quantized.shape
    assert (quantized == quantizer.indices_to_codes(indices)).all()
