import unittest

import torch
from torch.nn import functional as F

from model_bilstm_window_v2_2 import EXPECTED_PARAMETER_COUNTS, MDSoPBiLSTMRegressor


class TestModelWindow(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_exact_parameter_counts(self):
        for window in (1, 15):
            model = MDSoPBiLSTMRegressor(window_size=window)
            self.assertEqual(model.parameter_count(), EXPECTED_PARAMETER_COUNTS[window])

    def test_finite_forward_backward_both_heads(self):
        x = torch.randn(2, 9, 1182)
        lengths = torch.tensor([9, 5])
        residue_mask = torch.arange(9).unsqueeze(0) < lengths.unsqueeze(1)
        for window in (1, 15):
            model = MDSoPBiLSTMRegressor(window_size=window)
            prediction = model(x, lengths, residue_mask)
            self.assertEqual(tuple(prediction.shape), (2, 9))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(torch.equal(prediction[1, 5:], torch.zeros(4)))
            loss = prediction[residue_mask].square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_conv15_matches_explicit_window_expansion(self):
        model = MDSoPBiLSTMRegressor(window_size=15).eval()
        hidden = torch.randn(2, 11, 512)
        mask = torch.ones(2, 11, dtype=torch.bool)
        with torch.no_grad():
            actual = model.forward_from_hidden(hidden, mask)
            conv = model.window_projection
            output = model.output_layer
            padded = F.pad(hidden.transpose(1, 2), (7, 7))
            windows = padded.unfold(2, 15, 1)  # [B,C,L,K]
            projected = torch.einsum("bclk,ock->bol", windows, conv.weight)
            projected = projected + conv.bias.view(1, -1, 1)
            activated = F.gelu(projected)
            explicit = torch.einsum("bcl,oc->bol", activated, output.weight[:, :, 0])
            explicit = explicit + output.bias.view(1, -1, 1)
            explicit = explicit.squeeze(1)
        self.assertTrue(torch.allclose(actual, explicit, atol=1e-6, rtol=1e-5))

    def test_window_center_and_zero_chain_edges(self):
        model = MDSoPBiLSTMRegressor(window_size=15).eval()
        with torch.no_grad():
            model.window_projection.weight.zero_()
            model.window_projection.bias.zero_()
            model.output_layer.weight.zero_()
            model.output_layer.bias.zero_()
            model.window_projection.weight[0, 0, 7] = 1.0
            model.output_layer.weight[0, 0, 0] = 1.0
        hidden = torch.zeros(1, 5, 512)
        hidden[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones(1, 5, dtype=torch.bool)
        with torch.no_grad():
            actual = model.forward_from_hidden(hidden, mask)
        self.assertTrue(torch.allclose(actual[0], F.gelu(hidden[0, :, 0]), atol=1e-7))

        with torch.no_grad():
            model.window_projection.weight.zero_()
            model.window_projection.weight[0, 0, :] = 1.0
        with torch.no_grad():
            edge_output = model.forward_from_hidden(hidden, mask)
        expected_sum = hidden[0, :, 0].sum().expand(5)
        self.assertTrue(torch.allclose(edge_output[0], F.gelu(expected_sum), atol=1e-6))

    def test_single_vs_mixed_length_batch_invariance(self):
        model = MDSoPBiLSTMRegressor(window_size=15).eval()
        short = torch.randn(1, 6, 1182)
        long = torch.randn(1, 13, 1182)
        with torch.no_grad():
            alone = model(short, torch.tensor([6]))[0, :6]
            mixed_x = torch.zeros(2, 13, 1182)
            mixed_x[0, :6] = short[0]
            mixed_x[1] = long[0]
            mixed = model(mixed_x, torch.tensor([6, 13]))[0, :6]
        self.assertTrue(torch.allclose(alone, mixed, atol=2e-6, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()

