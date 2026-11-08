import unittest

import torch
from torch.nn import functional as F

from mdsop_loss_v2_2 import (
    LossAccessError,
    NoSupervisionError,
    mixed_source_record_macro_smooth_l1,
    record_macro_smooth_l1,
    training_loss_from_batch,
    validation_loss_from_batch,
)


class TestMasksAndLoss(unittest.TestCase):
    def test_invalid_nan_labels_never_enter_arithmetic(self):
        prediction = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        target = torch.tensor([[0.0, float("nan"), 0.4]])
        mask = torch.tensor([[True, False, True]])
        first = record_macro_smooth_l1(prediction, target, mask).loss
        changed = target.clone()
        changed[0, 1] = -99999.0
        second = record_macro_smooth_l1(prediction, changed, mask).loss
        self.assertTrue(torch.isfinite(first))
        self.assertTrue(torch.equal(first, second))
        first.backward()
        self.assertEqual(float(prediction.grad[0, 1]), 0.0)

    def test_changing_valid_label_changes_loss(self):
        prediction = torch.zeros(1, 2)
        target = torch.zeros(1, 2)
        mask = torch.tensor([[True, False]])
        baseline = record_macro_smooth_l1(prediction, target, mask).loss
        target[0, 0] = 1.0
        changed = record_macro_smooth_l1(prediction, target, mask).loss
        self.assertGreater(float(changed), float(baseline))

    def test_record_macro_is_not_residue_pooling(self):
        prediction = torch.zeros(2, 3)
        target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, False], [True, True, True]])
        result = record_macro_smooth_l1(prediction, target, mask)
        pooled = F.smooth_l1_loss(prediction[mask], target[mask], beta=0.05, reduction="mean")
        self.assertFalse(torch.allclose(result.loss, pooled))
        self.assertTrue(torch.allclose(result.loss, result.record_losses.mean()))
        self.assertEqual(result.valid_counts.tolist(), [1, 3])

    def test_zero_label_record_is_skipped_and_all_zero_raises(self):
        prediction = torch.zeros(2, 2)
        target = torch.zeros(2, 2)
        mask = torch.tensor([[True, False], [False, False]])
        result = record_macro_smooth_l1(prediction, target, mask)
        self.assertEqual(result.record_indices.tolist(), [0])
        with self.assertRaises(NoSupervisionError):
            record_macro_smooth_l1(prediction, target, torch.zeros_like(mask))

    def test_mixed_loss_is_half_each_source_macro(self):
        prediction = torch.zeros(3, 2)
        target = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
        mask = torch.ones(3, 2, dtype=torch.bool)
        result = mixed_source_record_macro_smooth_l1(
            prediction, target, mask, ["md_ired", "soppe_legacy", "soppe_legacy"]
        )
        expected = 0.5 * result.md_loss.loss + 0.5 * result.soppe_loss.loss
        self.assertTrue(torch.equal(result.loss, expected))

    def test_role_guards_and_validation_no_grad(self):
        prediction = torch.zeros(1, 2, requires_grad=True)
        base = {
            "y": torch.ones(1, 2),
            "loss_mask": torch.ones(1, 2, dtype=torch.bool),
            "eval_mask": torch.ones(1, 2, dtype=torch.bool),
            "sources": ["md_ired"],
        }
        with self.assertRaises(LossAccessError):
            training_loss_from_batch(prediction, {**base, "roles": ["validation"]}, mixed_sources=False)
        validation = validation_loss_from_batch(
            prediction, {**base, "roles": ["validation"]}
        )
        self.assertFalse(validation.requires_grad)


if __name__ == "__main__":
    unittest.main()

