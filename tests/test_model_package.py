import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mercurion.model_package import ModelPackageError, checksum, publish_package, validate_package
from mercurion.model_spec import TOP4_LABELS


class PackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"synthetic checkpoint; loading tensors is tested separately")
        self.output = self.root / "package"
        self.arguments = {"thresholds": {"global": 0.5, "per_label": dict.fromkeys(TOP4_LABELS, 0.4)},
                          "model_version": "synthetic-v1", "export_revision": "a" * 40,
                          "calibration": {"split": "validation", "samples": 8,
                                          "features_sha256": "b" * 64, "targets_sha256": "c" * 64},
                          "libraries": {"torch": "test", "numpy": "test", "rdkit": "test"}}

    def publish(self):
        return publish_package(self.checkpoint, self.output, **self.arguments)

    def test_roundtrip_preserves_snapshot_and_records_identity(self):
        self.publish()
        manifest, thresholds = validate_package(self.output)
        self.assertEqual(manifest["model_version"], "synthetic-v1")
        self.assertEqual(manifest["files"]["weights.pt"], checksum(self.checkpoint))
        self.assertEqual((self.output / "weights.pt").read_bytes(), self.checkpoint.read_bytes())
        self.assertEqual(thresholds, self.arguments["thresholds"])

    def test_incomplete_and_unknown_schema_packages_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(ModelPackageError):
            validate_package(self.output)
        self.output.rmdir()
        self.publish()
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["schema_version"] = 2
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ModelPackageError):
            validate_package(self.output)

    def test_checksum_catches_modified_weights(self):
        self.publish()
        (self.output / "weights.pt").write_bytes(b"different checkpoint")
        with self.assertRaises(ModelPackageError):
            validate_package(self.output)

    def test_hash_valid_but_incompatible_metadata_is_rejected(self):
        self.publish()
        original = json.loads((self.output / "manifest.json").read_text())
        cases = {"model": {"input_dim": 1}, "labels": ["wrong"],
                 "fingerprint": {"algorithm": "other"}, "export_revision": "not-a-sha",
                 "calibration": {"split": "test"}, "files": {"../weights.pt": "a" * 64}}
        for field, value in cases.items():
            with self.subTest(field=field):
                (self.output / "manifest.json").write_text(json.dumps({**original, field: value}))
                with self.assertRaises(ModelPackageError):
                    validate_package(self.output)

    def test_nonfinite_missing_and_invalid_thresholds_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            with self.subTest(value=value):
                self.arguments["thresholds"]["global"] = value
                with self.assertRaises(ModelPackageError):
                    self.publish()
                self.assertFalse(self.output.exists())
        self.arguments["thresholds"]["global"] = 0.5
        self.arguments["thresholds"]["per_label"].pop(TOP4_LABELS[0])
        with self.assertRaises(ModelPackageError):
            self.publish()

    def test_invalid_threshold_json_is_rejected_even_with_updated_hash(self):
        self.publish()
        path = self.output / "thresholds.json"
        path.write_text('{"global": NaN, "per_label": {}}')
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["files"]["thresholds.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ModelPackageError):
            validate_package(self.output)

    def test_existing_package_is_never_overwritten(self):
        self.publish()
        before = checksum(self.output / "manifest.json")
        with self.assertRaises(ModelPackageError):
            self.publish()
        self.assertEqual(checksum(self.output / "manifest.json"), before)

    def test_failed_publication_cleans_staging_and_exposes_no_partial_package(self):
        with patch("mercurion.model_package.shutil.copyfile", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".model-package-*")), [])


if __name__ == "__main__":
    unittest.main()
