import pytest
import torch

from vector_quantize_pytorch.finite_scalar_quantization import FSQ


class TestFSQ:
    levels = [8,5,5,5]
    quantizer = FSQ(levels)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, 1024, 4) # 4 since there are 4 levels
        quantized, indices = self.quantizer(features)

        assert quantized.shape == features.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestFSQNoIndices:
    levels = [8,5,5,5]
    quantizer = FSQ(levels, return_indices=False)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, 1024, 4) # 4 since there are 4 levels
        _, indices = self.quantizer(features)

        assert indices is None

class TestFSQWithDim:
    levels = [8,5,5,5]
    dim = 16
    quantizer = FSQ(levels, dim=dim)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        image_feats = torch.randn(1, 16, 32, 32)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
