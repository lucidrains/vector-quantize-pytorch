import torch

from vector_quantize_pytorch.latent_quantization import LatentQuantize


class TestLatentQuantizer:
    quantizer = LatentQuantize(
        levels=[5, 5, 8],  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    quantizer_same_level = LatentQuantize(
        levels=[5, 5, 5],  # number of levels per codebook dimension
        dim=16,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    def test_init(self):
        assert self.quantizer

    def test_init_same_level(self):
        assert self.quantizer_same_level

    def test_forward_shape(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()

    def test_forward_shape_same_level(self):
        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer_same_level(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()
