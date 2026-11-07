import copy
import math
import unittest

import torch

from api.inference import predict


class KnownLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("logits", torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, math.log(3), 0, -math.log(3), math.log(9)
        ], dtype=torch.float32).reshape(1, 12))
        self.grad_enabled = None
        self.input_signature = None

    def forward(self, features):
        self.grad_enabled = torch.is_grad_enabled()
        self.input_signature = (features.dtype, tuple(features.shape))
        return self.logits + self.marker * 0


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = KnownLogits().eval()
        self.thresholds = {"SR-ATAD5": 0.7, "NR-AhR": 0.5, "SR-MMP": 0.2, "SR-p53": 0.95}

    def test_known_logits_map_to_named_labels_with_sigmoid_and_strict_threshold(self):
        result = predict("CCO", self.model, "cpu", thresholds=self.thresholds)
        expected = {"SR-ATAD5": (0.75, True), "NR-AhR": (0.5, False),
                    "SR-MMP": (0.25, True), "SR-p53": (0.9, False)}
        self.assertEqual(set(result), set(expected))
        for label, (probability, positive) in expected.items():
            self.assertAlmostEqual(result[label]["probability"], probability, places=6)
            self.assertEqual(result[label]["threshold"], self.thresholds[label])
            self.assertIs(result[label]["is_positive"], positive)

    def test_prediction_has_no_gradients_or_state_mutation(self):
        state = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        thresholds = copy.deepcopy(self.thresholds)
        first = predict("CCO", self.model, "cpu", thresholds=self.thresholds)
        self.assertEqual(first, predict("CCO", self.model, "cpu", thresholds=self.thresholds))
        self.assertFalse(self.model.grad_enabled)
        self.assertEqual(self.model.input_signature, (torch.float32, (1, 2048)))
        self.assertIsNone(self.model.marker.grad)
        for name, tensor in state.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[name]))
        self.assertEqual(thresholds, self.thresholds)

    def test_nonfinite_logits_and_wrong_output_shapes_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.model.logits[0, 2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict("CCO", self.model, "cpu", thresholds=self.thresholds)
        for shape in ((1, 11), (2, 12), (12,)):
            self.model.logits = torch.zeros(shape)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                predict("CCO", self.model, "cpu", thresholds=self.thresholds)

    def test_invalid_legacy_threshold_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict("CCO", self.model, "cpu", thresholds={**self.thresholds, "NR-AhR": value})


if __name__ == "__main__":
    unittest.main()
