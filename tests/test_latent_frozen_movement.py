import unittest

import torch

from vector_quantize_pytorch import LatentQuantize


class TestLatentFrozenMovement(unittest.TestCase):
    def test_frozen_scalar_values_follow_dtype_and_device_movement(self):
        model = LatentQuantize(levels=[3, 4], dim=2, optimize_values=False).double()
        self.assertTrue(all(value.dtype == torch.float64 for value in model.values_per_latent))
        self.assertTrue(all(not value.requires_grad for value in model.values_per_latent))
        x = torch.tensor([[[.1, -.2], [.2, -.1]]], dtype=torch.float64, requires_grad=True)
        output, _, loss = model(x)
        self.assertEqual(output.dtype, torch.float64)
        (output.sum() + loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        model.to("meta")
        self.assertTrue(all(value.device.type == "meta" for value in model.values_per_latent))

    def test_frozen_values_roundtrip_as_module_state(self):
        first = LatentQuantize(levels=[3, 4], dim=2, optimize_values=False)
        second = LatentQuantize(levels=[3, 4], dim=2, optimize_values=False)
        with torch.no_grad():
            first.values_per_latent[0].add_(.125)
        second.load_state_dict(first.state_dict())
        torch.testing.assert_close(second.values_per_latent[0], first.values_per_latent[0])
        self.assertTrue(all(not value.requires_grad for value in second.values_per_latent))


    def test_legacy_frozen_checkpoint_loads_strictly_with_initialized_tables(self):
        old_model = LatentQuantize(levels=[3, 4], dim=5, optimize_values=False)
        legacy_state = old_model.state_dict()
        for key in list(legacy_state):
            if key.startswith("values_per_latent."):
                del legacy_state[key]
        legacy_state._metadata[""]["version"] = 1
        restored = LatentQuantize(levels=[3, 4], dim=5, optimize_values=False)
        restored.load_state_dict(legacy_state, strict=True)
        for actual, expected in zip(restored.values_per_latent, old_model.values_per_latent):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(restored.project_out.weight, old_model.project_out.weight)

    def test_modern_missing_table_still_fails_strict_loading(self):
        model = LatentQuantize(levels=[3, 4], dim=2, optimize_values=False)
        state = model.state_dict()
        del state["values_per_latent.0"]
        with self.assertRaisesRegex(RuntimeError, "values_per_latent.0"):
            model.load_state_dict(state, strict=True)
    def test_runtime_freezing_does_not_change_legacy_learned_schema(self):
        model = LatentQuantize(levels=[3, 4], dim=2, optimize_values=True)
        model.requires_grad_(False)
        state = model.state_dict()
        state._metadata[""]["version"] = 1
        del state["values_per_latent.0"]
        with self.assertRaisesRegex(RuntimeError, "values_per_latent.0"):
            model.load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
