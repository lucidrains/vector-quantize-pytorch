import pytest
import torch
from einops import rearrange

from torch_einops_utils.torch_einops_utils import lens_to_mask

from vector_quantize_pytorch import HierarchicalVQ, VectorQuantize


# spatial dims, input shape, scales, level shapes, kwargs

SPATIAL_CASES = [
    pytest.param(
        1,
        (2, 32, 8),
        (1, 2, 4, 8),
        ((1,), (2,), (4,), (8,)),
        dict(),
        id = '1d'
    ),
    pytest.param(
        2,
        (2, 32, 6, 8),
        ((1, 2), (3, 4), (6, 8)),
        ((1, 2), (3, 4), (6, 8)),
        {'accept_image_fmap': True},
        id = '2d'
    ),
    pytest.param(
        3,
        (2, 32, 2, 4, 4),
        ((1, 1, 1), (2, 2, 2), (2, 4, 4)),
        ((1, 1, 1), (2, 2, 2), (2, 4, 4)),
        {'accept_3d_fmap': True},
        id = '3d'
    )
]


@pytest.mark.parametrize('channel_first', (True, False), ids = ('channel-first', 'channel-last'))
@pytest.mark.parametrize('shared_codebook', (True, False), ids = ('shared-codebook', 'per-level-codebooks'))
@pytest.mark.parametrize(
    'spatial_dims, input_shape, scales, level_shapes, fmap_kwargs',
    SPATIAL_CASES
)
def test_hierarchical_vq_round_trip(
    channel_first,
    shared_codebook,
    spatial_dims,
    input_shape,
    scales,
    level_shapes,
    fmap_kwargs
):
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = scales,
        shared_codebook = shared_codebook,
        channel_first = channel_first,
        kmeans_init = False,
        quant_resi = 0.5,
        **fmap_kwargs
    )

    assert hq.spatial_dims == spatial_dims

    if shared_codebook:
        assert hq.vq is not None
        assert hq.vqs is None
    else:
        assert hq.vq is None
        assert len(hq.vqs) == len(level_shapes)

    x = torch.randn(input_shape)

    if not channel_first:
        x = rearrange(x, 'b c ... -> b ... c')

    quantized, indices, commit_loss = hq(x)

    assert quantized.shape == x.shape
    assert len(indices) == len(level_shapes)
    assert torch.isfinite(commit_loss)

    for level_indices, level_shape in zip(indices, level_shapes):
        assert level_indices.shape == (input_shape[0], *level_shape)

    reconstructed = hq.get_output_from_indices(indices)
    assert reconstructed.shape == x.shape

    reconstructed = hq.get_output_from_indices(indices, full_shape = input_shape[2:])
    assert reconstructed.shape == x.shape


def test_hierarchical_vq_text_embeddings_with_variable_length_mask():
    vocab_size, dim, seq_len = 128, 32, 8

    embed = torch.nn.Embedding(vocab_size, dim)

    hq = HierarchicalVQ(
        dim = dim,
        codebook_size = 16,
        scales = (1, 2, 4, 8),
        channel_first = False,
        kmeans_init = False
    )

    token_ids = torch.randint(0, vocab_size, (2, seq_len))
    lens = torch.tensor([8, 5])
    mask = lens_to_mask(lens, seq_len)

    x = embed(token_ids)
    quantized, indices, commit_loss = hq(x, mask = mask)

    assert x.shape == (2, seq_len, dim)
    assert quantized.shape == x.shape
    assert indices[-1].shape == (2, seq_len)
    assert (indices[-1][1, 5:] == -1).all()
    assert torch.isfinite(commit_loss)

    reconstructed = hq.get_output_from_indices(indices)
    assert reconstructed.shape == x.shape


@pytest.mark.parametrize(
    'spatial_dims, input_shape, scales, level_shapes, fmap_kwargs',
    SPATIAL_CASES
)
def test_hierarchical_vq_mask_marks_padding_with_minus_one(
    spatial_dims,
    input_shape,
    scales,
    level_shapes,
    fmap_kwargs
):
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = scales,
        kmeans_init = False,
        **fmap_kwargs
    )

    x = torch.randn(input_shape)
    spatial_shape = input_shape[2:]
    mask = torch.ones((input_shape[0], *spatial_shape), dtype = torch.bool)

    half = spatial_shape[-1] // 2
    mask[1, ..., half:] = False

    quantized, indices, _ = hq(x, mask = mask)

    assert quantized.shape == x.shape
    assert hq.spatial_dims == spatial_dims
    assert indices[-1].shape == (input_shape[0], *level_shapes[-1])
    assert (indices[-1][1, ..., half:] == -1).all()

    reconstructed = hq.get_output_from_indices(indices)
    assert reconstructed.shape == x.shape


@pytest.mark.parametrize('channel_first', (True, False), ids = ('channel-first', 'channel-last'))
def test_hierarchical_vq_lens_equivalent_to_mask(channel_first):
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = (1, 2, 4, 8),
        channel_first = channel_first,
        kmeans_init = False
    ).eval()

    x = torch.randn(2, 32, 8)

    if not channel_first:
        x = rearrange(x, 'b c l -> b l c')

    lens = torch.tensor([8, 5])
    mask = lens_to_mask(lens, 8)

    quantized_lens, indices_lens, _ = hq(x, lens = lens)
    quantized_mask, indices_mask, _ = hq(x, mask = mask)

    assert quantized_lens.shape == x.shape
    assert torch.allclose(quantized_lens, quantized_mask)

    for level_indices_lens, level_indices_mask in zip(indices_lens, indices_mask):
        assert (level_indices_lens == level_indices_mask).all()

    assert (indices_lens[-1][1, 5:] == -1).all()


def test_hierarchical_vq_per_level_codebook_sizes():
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = (8, 16, 32),
        scales = (1, 2, 4),
        shared_codebook = False,
        kmeans_init = False
    )

    assert hq.codebook_sizes == (8, 16, 32)
    assert tuple(vq.codebook_size for vq in hq.vqs) == (8, 16, 32)

    quantized, indices, _ = hq(torch.randn(2, 32, 4))

    assert quantized.shape == (2, 32, 4)
    assert len(indices) == 3

    # tied codebook requires uniform sizes

    with pytest.raises(AssertionError, match = 'shared codebook'):
        HierarchicalVQ(
            dim = 32,
            codebook_size = (8, 16),
            scales = (1, 2),
            shared_codebook = True,
            kmeans_init = False
        )


def test_hierarchical_vq_nested_dropout_cuts_at_a_level():
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = (1, 2, 4, 8),
        nested_dropout = 1.,
        nested_dropout_min_levels = 2,
        kmeans_init = False
    )

    hq.train()
    x = torch.randn(2, 32, 8)
    quantized, indices, commit_loss, cutoff = hq(x, return_cutoff = True)

    assert cutoff is not None
    assert 1 <= cutoff <= 2
    assert quantized.shape == x.shape
    assert len(indices) == 4
    assert torch.isfinite(commit_loss)

    hq.eval()
    quantized, _, commit_loss, cutoff = hq(x, return_cutoff = True)

    assert cutoff is None
    assert quantized.shape == x.shape
    assert torch.isfinite(commit_loss)


@pytest.mark.parametrize(
    'sync_codebook',
    (True, False, None),
    ids = ('sync', 'local', 'default')
)
def test_hierarchical_vq_sync_codebook_matches_vector_quantize(sync_codebook):
    reference_vq = VectorQuantize(
        dim = 32,
        codebook_size = 16,
        sync_codebook = sync_codebook,
        kmeans_init = False
    )

    expected_use_ddp = reference_vq._codebook.use_ddp

    per_level_hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = (1, 2, 4),
        shared_codebook = False,
        sync_codebook = sync_codebook,
        kmeans_init = False
    )

    shared_hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = (1, 2, 4),
        shared_codebook = True,
        sync_codebook = sync_codebook,
        kmeans_init = False
    )

    assert all(vq._codebook.use_ddp == expected_use_ddp for vq in per_level_hq.vqs)
    assert shared_hq.vq._codebook.use_ddp == expected_use_ddp


def test_hierarchical_vq_matches_vector_quantize_with_one_level():
    hq = HierarchicalVQ(
        dim = 32,
        codebook_size = 16,
        scales = (8,),
        shared_codebook = True,
        quant_resi = 0.,
        kmeans_init = False
    ).eval()

    vq = VectorQuantize(
        dim = 32,
        codebook_size = 16,
        kmeans_init = False
    ).eval()

    vq.load_state_dict(hq.vq.state_dict())

    x = torch.randn(2, 32, 8)

    hq_quantized, hq_indices, _ = hq(x)
    vq_quantized, vq_indices, _ = vq(rearrange(x, 'b c l -> b l c'))

    assert torch.allclose(hq_quantized, rearrange(vq_quantized, 'b l c -> b c l'))
    assert (hq_indices[0] == vq_indices).all()

    hq_reconstructed = hq.get_output_from_indices(hq_indices)
    vq_reconstructed = vq.get_output_from_indices(vq_indices)

    assert torch.allclose(hq_reconstructed, rearrange(vq_reconstructed, 'b l c -> b c l'))
