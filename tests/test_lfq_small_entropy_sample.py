import unittest

import torch

from vector_quantize_pytorch import LFQ


class TestLFQSmallEntropySample(unittest.TestCase):
    def test_single_token_retains_its_entropy(self):
        full = LFQ(dim=2, codebook_size=4, entropy_loss_weight=1., diversity_gamma=.7)
        sampled = LFQ(dim=2, codebook_size=4, entropy_loss_weight=1., diversity_gamma=.7, frac_per_sample_entropy=.1)
        x = torch.tensor([[[.1, -.2]]])
        expected = full(x, inv_temperature=1.).entropy_aux_loss
        actual = sampled(x, inv_temperature=1.).entropy_aux_loss
        self.assertTrue(torch.isfinite(actual))
        torch.testing.assert_close(actual, expected)

    def test_small_nonempty_batch_has_finite_loss_and_gradients(self):
        model = LFQ(dim=2, codebook_size=4, entropy_loss_weight=1., diversity_gamma=.7, frac_per_sample_entropy=.2)
        x = torch.tensor([[[.1, -.2], [.2, -.1]]], requires_grad=True)
        loss = model(x, inv_temperature=1.).entropy_aux_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
