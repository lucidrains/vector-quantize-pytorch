import unittest

import torch

from vector_quantize_pytorch import LFQ


def reference_entropy(x, model, mask, temperature):
    values = x.reshape(*x.shape[:2], model.num_codebooks, model.codebook_dim)[mask]
    distances = ((values[:, :, None, :] - model.codebook[None, None, :, :]) ** 2).sum(-1)
    probabilities = (-temperature * distances).softmax(-1)
    sample_entropy = -(probabilities * probabilities.log()).sum(-1).mean()
    average = probabilities.mean(0)
    batch_entropy = -(average * average.log()).sum(-1).mean()
    return sample_entropy - model.diversity_gamma * batch_entropy


class TestLFQMaskedEntropy(unittest.TestCase):
    def test_all_valid_mask_preserves_unmasked_entropy(self):
        model = LFQ(dim=2, codebook_size=4, entropy_loss_weight=1.)
        x = torch.tensor([[[.1, -.2], [.3, .2]], [[-.2, .3], [-.1, -.2]]])
        actual = model(x, mask=torch.ones(2, 2, dtype=torch.bool), inv_temperature=1.).entropy_aux_loss
        expected = model(x, inv_temperature=1.).entropy_aux_loss
        torch.testing.assert_close(actual, expected)

    def test_selected_tokens_and_multiple_codebooks_match_shannon_entropy(self):
        for codebooks in (1, 2):
            with self.subTest(codebooks=codebooks):
                model = LFQ(
                    dim=2 * codebooks, codebook_size=4, num_codebooks=codebooks,
                    entropy_loss_weight=1., diversity_gamma=.7,
                )
                x = torch.linspace(-.3, .3, 6 * 2 * codebooks).reshape(2, 3, 2 * codebooks)
                mask = torch.tensor([[True, False, True], [False, True, True]])
                actual = model(x, mask=mask, inv_temperature=1.).entropy_aux_loss
                torch.testing.assert_close(actual, reference_entropy(x, model, mask, 1.))

    def test_entropy_gradients_match_selected_distribution(self):
        model = LFQ(dim=4, codebook_size=4, num_codebooks=2, entropy_loss_weight=1., diversity_gamma=.7)
        x = torch.linspace(-.3, .3, 24).reshape(2, 3, 4).requires_grad_()
        mask = torch.tensor([[True, False, True], [False, True, True]])
        actual = torch.autograd.grad(model(x, mask=mask, inv_temperature=1.).entropy_aux_loss, x)[0]
        expected = torch.autograd.grad(reference_entropy(x, model, mask, 1.), x)[0]
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
