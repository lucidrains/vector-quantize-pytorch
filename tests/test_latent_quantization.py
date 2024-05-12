from vector_quantize_pytorch.latent_quantization import LatentQuantize
import torch

class TestLatentQuantizer:

    quantizer = LatentQuantize(
        levels = [5, 5, 8],      # number of levels per codebook dimension
        dim = 16,                   # input dim
        commitment_loss_weight=0.1,  
        quantization_loss_weight=0.1,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward_shape(self):

        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, _ = self.quantizer(image_feats)

        assert image_feats.shape == quantized.shape
        assert (quantized == self.quantizer.indices_to_codes(indices)).all()