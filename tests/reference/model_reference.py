"""Explicit real-package regression; never silently skipped when a package is missing."""
import json
import os
from pathlib import Path
import unittest

from mercurion.model_package import checksum
from runtime import load_inference_runtime


class ModelReferenceTests(unittest.TestCase):
    def test_real_package_matches_pre_task_cpu_predictions(self):
        selected = os.environ.get("MODEL_PACKAGE_DIR")
        if not selected:
            raise RuntimeError("The reference group requires MODEL_PACKAGE_DIR; see docs/application-testing.md")
        root = Path(selected)
        fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/model-reference.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["fixture_version"], 1)
        self.assertEqual(len(fixture["cases"]), 4)
        self.assertEqual(checksum(root / "weights.pt"), fixture["weights_sha256"])
        self.assertEqual(checksum(root / "thresholds.json"), fixture["thresholds_sha256"])
        model, device, predict = load_inference_runtime(package_dir=root)
        self.assertEqual(device, "cpu", "Reference fixture is validated for CPU")
        for case in fixture["cases"]:
            actual = predict(case["smiles"], model, device)
            self.assertEqual(set(actual), set(case["response"]))
            for label, expected in case["response"].items():
                with self.subTest(smiles=case["smiles"], label=label):
                    self.assertAlmostEqual(actual[label]["probability"], expected["probability"],
                                           delta=fixture["absolute_tolerance"])
                    self.assertEqual(actual[label]["threshold"], expected["threshold"])
                    self.assertIs(actual[label]["is_positive"], expected["is_positive"])


if __name__ == "__main__":
    unittest.main()
