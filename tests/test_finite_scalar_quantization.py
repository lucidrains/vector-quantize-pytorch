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

    def test_forward_images(self):
        image_feats = torch.randn(1, 32, 32, 4)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
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

    def test_forward_images(self):
        image_feats = torch.randn(1, 32, 32, 4)
        _, indices = self.quantizer(image_feats)

        assert indices is None

class TestFSQWithDimAndChannelFirst:
    levels = [8,5,5,5]
    dim = 16
    quantizer = FSQ(levels, dim=dim, channel_first=True)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, self.dim, 1024) # 4 since there are 4 levels
        quantized, indices = self.quantizer(features)

        assert quantized.shape == features.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


    def test_forward_images(self):
        image_feats = torch.randn(1, self.dim, 32, 32)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))

class TestFSQSeveralCodebooks:
    levels = [8,5,5,5]
    quantizer = FSQ(levels, num_codebooks=2)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, 1024, 8) # 4 since there are 4 levels
        quantized, indices = self.quantizer(features)

        assert quantized.shape == features.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))

    def test_forward_images(self):
        image_feats = torch.randn(1, 32, 32, 8)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))

class TestFSQSeveralCodebooksKeepCodebooks:
    levels = [8,5,5,5]
    num_codebooks = 2
    quantizer = FSQ(levels, num_codebooks=num_codebooks, keep_num_codebooks_dim=True)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, 1024, 8) # 4 since there are 4 levels
        quantized, indices = self.quantizer(features)

        assert quantized.shape == features.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
        assert indices.shape[-1] == self.num_codebooks

    def test_forward_images(self):
        image_feats = torch.randn(1, 32, 32, 8)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
        assert indices.shape[-1] == self.num_codebooks


class TestFSQWithDimAndChannelFirstSeveralCodebooks:
    levels = [8,5,5,5]
    dim = 16
    quantizer = FSQ(levels, dim=dim, channel_first=True, num_codebooks=2)

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        features = torch.randn(1, self.dim, 1024) # 4 since there are 4 levels
        quantized, indices = self.quantizer(features)

        assert quantized.shape == features.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


    def test_forward_images(self):
        image_feats = torch.randn(1, self.dim, 32, 32)
        quantized, indices = self.quantizer(image_feats)

        assert quantized.shape == image_feats.shape
        assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
