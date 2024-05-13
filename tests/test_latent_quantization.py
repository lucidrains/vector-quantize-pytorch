import pytest
import torch

from vector_quantize_pytorch.latent_quantization import LatentQuantize


class TestLatentQuantizer:
    quantizer = LatentQuantize(
        levels=[5, 5, 8],  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward_images(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()

    def test_forward_video(self):
        video_feats = torch.randn(1, 16, 10, 32, 32)

        quantized, indices, _ = self.quantizer(video_feats)

        assert video_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()

    def test_forward_series(self):
        series_feats = torch.randn(1, 16, 64)

        quantized, indices, _ = self.quantizer(series_feats)

        assert series_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()


class TestLatentQuantizerNoOptim:
    quantizer = LatentQuantize(
        levels=[5, 5, 8],  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
        optimize_values=False,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward_shape_int(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()


class TestLatentQuantizerSameLevel:
    quantizer_same_level = LatentQuantize(
        levels=[5, 5, 5],  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    def test_init_same_level(self):
        assert self.quantizer_same_level

    def test_forward_shape_same_level(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer_same_level(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer_same_level.indices_to_codes(indices)).all()


class TestLatentQuantizerInt:
    quantizer_int = LatentQuantize(
        levels=5,  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
        codebook_dim=3,
    )

    def test_init_int(self):
        assert self.quantizer_int

    def test_forward_shape_int(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer_int(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer_int.indices_to_codes(indices)).all()


class TestLatentQuantizerBadInt:
    with pytest.raises(RuntimeError):
        quantizer_int = LatentQuantize(
            levels=5,  # number of levels per codebook dimension
            dim=16,  # input dim
            commitment_loss_weight=0.1,
            quantization_loss_weight=0.1,
            # codebook_dim=3,
        )
