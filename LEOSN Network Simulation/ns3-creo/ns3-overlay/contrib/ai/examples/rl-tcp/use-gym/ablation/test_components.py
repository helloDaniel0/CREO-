#!/usr/bin/env python3
import unittest

import numpy as np
import torch

from ablation_agent import (
    VARIANTS,
    AblationAgent,
    daubechies_denoise,
    get_spec,
    pdpa_action_space,
)


class AblationComponentTest(unittest.TestCase):
    def test_dwt_preserves_constant_signal(self):
        signal = np.full(300, 37.0, dtype=np.float32)
        np.testing.assert_allclose(daubechies_denoise(signal), signal, atol=1e-5)

    def test_pdpa_produces_ordered_seven_actions(self):
        x = np.linspace(20.0, 60.0, 80) + 3.0 * np.sin(np.arange(80) / 4.0)
        actions, metadata = pdpa_action_space(x)
        self.assertEqual(len(actions), 7)
        self.assertEqual(actions[3], 1.0)
        self.assertTrue(all(actions[index] < actions[index + 1] for index in range(6)))
        self.assertGreater(metadata["coverage_probability"], 0.0)

    def test_every_variant_has_actor_and_two_critics(self):
        batch_size = 2
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                agent = AblationAgent(
                    get_spec(variant),
                    [0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5],
                    capacity_window=32,
                    history_len=10,
                    batch_size=2,
                    device="cpu",
                )
                batch = {
                    "capacity": torch.zeros(batch_size, 32),
                    "trend": torch.zeros(batch_size, 32),
                    "fluct": torch.zeros(batch_size, 32),
                    "metrics": torch.zeros(batch_size, 10, 9),
                }
                outputs = agent.net(batch)
                self.assertEqual(len(outputs), 3)
                for output in outputs:
                    self.assertEqual(tuple(output.shape), (batch_size, 7))


if __name__ == "__main__":
    unittest.main()
