import tempfile
import unittest
from pathlib import Path

import numpy as np

from mdsop_training_v2_2 import (
    CyclingShuffleStream,
    atomic_write_text,
    save_training_curves,
    summarize_validation_records,
)


class TestTrainerUtilities(unittest.TestCase):
    def test_cycling_stream_never_drops_tail(self):
        stream = CyclingShuffleStream(5, 123)
        values = stream.next_indices(12)
        self.assertEqual(len(values), 12)
        self.assertEqual(sorted(values[:5]), list(range(5)))
        self.assertEqual(sorted(values[5:10]), list(range(5)))
        self.assertAlmostEqual(stream.equivalent_full_traversals, 12 / 5)

    def test_cycling_stream_resume_is_exact(self):
        stream = CyclingShuffleStream(7, 456)
        prefix = stream.next_indices(9)
        state = stream.state_dict()
        expected = stream.next_indices(25)
        restored = CyclingShuffleStream(7, 456)
        restored.load_state_dict(state)
        observed = restored.next_indices(25)
        self.assertEqual(expected, observed)
        self.assertEqual(prefix[:7], list(dict.fromkeys(prefix[:7])))

    def test_record_macro_differs_from_pooled_and_unique_sequence_is_reported(self):
        records = [
            {
                "record_id": "a",
                "sequence_sha256": "same",
                "prediction": np.array([1.0]),
                "target": np.array([0.0]),
            },
            {
                "record_id": "b",
                "sequence_sha256": "same",
                "prediction": np.zeros(3),
                "target": np.zeros(3),
            },
            {
                "record_id": "c",
                "sequence_sha256": "other",
                "prediction": np.array([0.2, 0.2]),
                "target": np.array([0.0, 0.0]),
            },
        ]
        summary = summarize_validation_records(records)
        self.assertEqual(summary["record_count"], 3)
        self.assertEqual(summary["unique_sequence_count"], 2)
        self.assertNotEqual(
            summary["protein_record_macro"]["rmse"], summary["pooled"]["rmse"]
        )

    def test_training_curve_artifacts_are_created(self):
        history = [
            {
                "virtual_epoch": 1,
                "train_loss": 0.5,
                "selection_metric": 0.4,
                "md_validation_macro_rmse": 0.4,
                "learning_rate": 1e-4,
            },
            {
                "virtual_epoch": 2,
                "train_loss": 0.3,
                "selection_metric": 0.35,
                "md_validation_macro_rmse": 0.35,
                "learning_rate": 1e-4,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png, svg = root / "curve.png", root / "curve.svg"
            save_training_curves(history, png, svg, "test")
            self.assertGreater(png.stat().st_size, 1000)
            self.assertGreater(svg.stat().st_size, 1000)

    def test_atomic_text_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            atomic_write_text(path, "first\n")
            atomic_write_text(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")


if __name__ == "__main__":
    unittest.main()
