from importlib.metadata import version
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from api.inference import smiles_to_fp
from mercurion.export_model import export_model
from mercurion.fingerprints import smiles_to_fingerprint
from mercurion.model import MercurionMLP
from mercurion.model_package import ModelPackageError, checksum, validate_package
from mercurion.model_spec import TOP4_LABELS
from runtime import load_inference_runtime


class ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.validation = self.root / "validation"
        self.validation.mkdir()
        self.features = np.zeros((8, 2048), dtype=np.uint8)
        self.targets = np.tile((np.arange(8) % 2)[:, None], (1, 12)).astype(np.float32)
        np.save(self.validation / "X_val.npy", self.features)
        np.save(self.validation / "y_val.npy", self.targets)
        self.model = MercurionMLP()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.model[-1].bias.copy_(torch.linspace(-1.2, 1.2, 12))
        self.checkpoint = self.root / "selected-checkpoint.pt"
        torch.save(self.model.state_dict(), self.checkpoint)
        self.model.eval()
        self.output = self.root / "package"

    def export(self):
        return export_model(self.checkpoint, self.validation, self.output,
                            model_version="synthetic-v1", export_revision="a" * 40)

    def test_exported_real_weights_load_in_runtime_and_predict_deterministically(self):
        self.export()
        with patch("torch.set_num_interop_threads"):
            model, device, predict = load_inference_runtime(package_dir=self.output)
        first = predict("CCO", model, device)
        second = predict("CCO", model, device)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(TOP4_LABELS))
        self.assertFalse(model.training)
        self.assertTrue(all(0 <= value["probability"] <= 1 for value in first.values()))
        self.assertEqual(predict("not-a-smiles", model, device), {"error": "Invalid SMILES"})

    def test_calibration_uses_selected_checkpoint_not_later_in_memory_model(self):
        selected_hash = checksum(self.checkpoint)
        with torch.no_grad():
            expected = torch.sigmoid(self.model(torch.as_tensor(self.features, dtype=torch.float32))).numpy()
            self.model.model[-1].bias.fill_(10)
        from mercurion.calibration import find_per_label_thresholds
        with patch("mercurion.export_model.find_per_label_thresholds", wraps=find_per_label_thresholds) as calibrate:
            self.export()
        np.testing.assert_allclose(calibrate.call_args.args[1], expected, atol=1e-7, rtol=1e-6)
        manifest, _ = validate_package(self.output)
        self.assertEqual(manifest["files"]["weights.pt"], selected_hash)
        self.assertEqual(checksum(self.checkpoint), selected_hash)

    def test_export_does_not_read_final_test_split_or_require_training_data(self):
        (self.validation / "X_test.npy").write_bytes(b"not-readable-as-numpy")
        (self.validation / "y_test.npy").write_bytes(b"not-readable-as-numpy")
        self.export()
        manifest, _ = validate_package(self.output)
        self.assertEqual(manifest["calibration"]["split"], "validation")
        self.assertEqual(manifest["calibration"]["samples"], 8)

    def test_incompatible_or_missing_validation_inputs_publish_nothing(self):
        np.save(self.validation / "X_val.npy", np.zeros((8, 4)))
        with self.assertRaises(ModelPackageError):
            self.export()
        self.assertFalse(self.output.exists())
        (self.validation / "X_val.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_configured_invalid_package_never_falls_back_to_legacy(self):
        self.output.mkdir()
        with patch("torch.set_num_interop_threads"), patch("api.inference.load_thresholds") as legacy:
            with self.assertRaises(ModelPackageError):
                load_inference_runtime(package_dir=self.output)
            legacy.assert_not_called()

    def test_shared_fingerprint_keeps_training_bits_and_runtime_dtype(self):
        for smiles in ("CCO", "C[C@H](O)F", "[13CH3]C"):
            with self.subTest(smiles=smiles):
                bits = smiles_to_fingerprint(smiles)
                runtime_bits = smiles_to_fp(smiles)
                self.assertEqual(bits.dtype, np.uint8)
                self.assertEqual(runtime_bits.dtype, np.float32)
                self.assertEqual(runtime_bits.shape, (2048,))
                np.testing.assert_array_equal(bits, runtime_bits)

    def test_hash_valid_but_incompatible_weights_fail_strict_state_loading(self):
        self.export()
        torch.save({"model.0.weight": torch.zeros((1, 1))}, self.output / "weights.pt")
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["weights.pt"] = checksum(self.output / "weights.pt")
        path.write_text(json.dumps(manifest))
        with patch("torch.set_num_interop_threads"), self.assertRaises(RuntimeError):
            load_inference_runtime(package_dir=self.output)

    def test_hash_valid_but_corrupt_archive_is_not_treated_as_loaded_model(self):
        self.export()
        (self.output / "weights.pt").write_bytes(b"not-a-torch-archive")
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["weights.pt"] = checksum(self.output / "weights.pt")
        path.write_text(json.dumps(manifest))
        with patch("torch.set_num_interop_threads"), self.assertRaises(pickle.UnpicklingError):
            load_inference_runtime(package_dir=self.output)

    def test_incompatible_library_version_fails_before_weight_loading(self):
        self.export()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["libraries"]["numpy"] = "0.0.0"
        path.write_text(json.dumps(manifest))
        with patch("torch.set_num_interop_threads"), patch("torch.load") as load:
            with self.assertRaises(ModelPackageError):
                load_inference_runtime(package_dir=self.output)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
