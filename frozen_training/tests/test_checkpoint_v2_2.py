import tempfile
import unittest
from pathlib import Path

import torch

from model_bilstm_window_v2_2 import (
    MDSoPBiLSTMRegressor,
    build_paired_initializations,
    load_model_checkpoint,
    save_model_checkpoint,
)


class TestCheckpoint(unittest.TestCase):
    def test_save_reload_prediction_identity_and_wrong_head_guard(self):
        torch.manual_seed(29)
        model = MDSoPBiLSTMRegressor(window_size=15).eval()
        x = torch.randn(2, 8, 1182)
        lengths = torch.tensor([8, 5])
        with torch.no_grad():
            expected = model(x, lengths)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_checkpoint(path, model, seed=29)
            loaded, _ = load_model_checkpoint(path, expected_window_size=15)
            loaded.eval()
            with torch.no_grad():
                observed = loaded(x, lengths)
            self.assertTrue(torch.equal(expected, observed))
            with self.assertRaisesRegex(ValueError, "expected 1"):
                load_model_checkpoint(path, expected_window_size=1)

    def test_paired_initializations_have_identical_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_paired_initializations(Path(directory), seed=20261007)
            w1 = manifest["checkpoints"]["w1"]
            w15 = manifest["checkpoints"]["w15"]
            self.assertEqual(w1["backbone_state_sha256"], w15["backbone_state_sha256"])
            self.assertEqual(
                set(manifest["route_mapping"].values()), {"w1", "w15"}
            )


if __name__ == "__main__":
    unittest.main()

