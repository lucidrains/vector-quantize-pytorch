import unittest

import torch

from vector_quantize_pytorch import LatentQuantize


def set_nonuniform_values(model):
    with torch.no_grad():
        model.values_per_latent[0].copy_(torch.tensor([-.73, -.11, .37]))
        model.values_per_latent[1].copy_(torch.tensor([-.59, -.29, .13, .81]))


class TestLatentLearnedCodec(unittest.TestCase):
    def test_all_indices_decode_current_nonuniform_scalar_values(self):
        model = LatentQuantize(levels=[3, 4], dim=2)
        set_nonuniform_values(model)
        indices = torch.arange(12).reshape(1, 12)
        actual = model.indices_to_codes(indices)
        expected = torch.stack((
            model.values_per_latent[0][indices % 3],
            model.values_per_latent[1][indices // 3],
        ), dim=1)
        torch.testing.assert_close(actual, expected)
        restored = model.codes_to_indices(actual.transpose(1, 2))
        self.assertTrue(torch.equal(restored, indices.to(torch.int32)))

    def test_native_forward_roundtrip_with_projection_and_multiple_codebooks(self):
        for codebooks in (1, 2):
            with self.subTest(codebooks=codebooks):
                torch.manual_seed(19)
                model = LatentQuantize(levels=[3, 4], dim=5, num_codebooks=codebooks).eval()
                set_nonuniform_values(model)
                x = torch.randn(2, 5, 3, 2)
                quantized, indices, _ = model(x)
                self.assertTrue(((indices >= 0) & (indices < 12)).all())
                torch.testing.assert_close(model.indices_to_codes(indices), quantized)

    def test_unsorted_scalar_codebooks_encode_by_nearest_value(self):
        model = LatentQuantize(levels=[3, 4], dim=2)
        with torch.no_grad():
            model.values_per_latent[0].copy_(torch.tensor([.37, -.73, -.11]))
            model.values_per_latent[1].copy_(torch.tensor([.81, -.59, .13, -.29]))
        codes = torch.tensor([[[.37, .81], [-.73, -.29], [-.11, .13]]])
        expected = torch.tensor([[0, 10, 8]], dtype=torch.int32)
        self.assertTrue(torch.equal(model.codes_to_indices(codes), expected))
        torch.testing.assert_close(model.indices_to_codes(expected).transpose(1, 2), codes)


if __name__ == "__main__":
    unittest.main()
