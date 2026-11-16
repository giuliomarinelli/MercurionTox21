"""Small, versioned inference package. Hashes detect drift, not a trusted signature."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from mercurion.model_spec import FINGERPRINT_SPEC, LABELS, MODEL_SPEC, TOP4_LABELS


class ModelPackageError(ValueError):
    pass


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def invalid_constant(value):
        raise ModelPackageError("Non-finite JSON value")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    except (OSError, ValueError, TypeError):
        raise ModelPackageError(f"Invalid package JSON: {path.name}") from None


def validate_thresholds(thresholds):
    if not isinstance(thresholds, dict) or set(thresholds) != {"global", "per_label"}:
        raise ModelPackageError("Missing global/per-label thresholds")
    values = thresholds["per_label"]
    if not isinstance(values, dict) or set(values) != set(TOP4_LABELS):
        raise ModelPackageError("Threshold labels do not match inference labels")
    for number in [thresholds["global"], *values.values()]:
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ModelPackageError("Thresholds must be finite probabilities")


def validate_package(directory):
    root = Path(directory)
    try:
        entries = list(root.iterdir())
        if {path.name for path in entries} != {"weights.pt", "thresholds.json", "manifest.json"}:
            raise ModelPackageError("Model package must contain exactly weights, thresholds and manifest")
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ModelPackageError("Model package entries must be regular files")
    except OSError:
        raise ModelPackageError("Model package directory is unreadable") from None
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ModelPackageError("Unsupported model package schema")
    if not isinstance(manifest.get("model_version"), str) or not manifest["model_version"].strip():
        raise ModelPackageError("Model version is required")
    if not isinstance(manifest.get("export_revision"), str) or not re.fullmatch(r"[a-fA-F0-9]{40}", manifest["export_revision"]):
        raise ModelPackageError("Export revision must be a full Git SHA")
    expected = {"model": MODEL_SPEC, "fingerprint": FINGERPRINT_SPEC, "labels": LABELS, "top_labels": TOP4_LABELS}
    for name, value in expected.items():
        if json.dumps(manifest.get(name), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ModelPackageError(f"Incompatible model package {name}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"weights.pt", "thresholds.json"}:
        raise ModelPackageError("Invalid model package file inventory")
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or checksum(root / name) != digest:
            raise ModelPackageError(f"Model package checksum mismatch: {name}")
    calibration = manifest.get("calibration")
    if (not isinstance(calibration, dict) or calibration.get("split") != "validation"
            or type(calibration.get("samples")) is not int or calibration["samples"] <= 0):
        raise ModelPackageError("Calibration must identify the validation split")
    for name in ("features_sha256", "targets_sha256"):
        if not isinstance(calibration.get(name), str) or not re.fullmatch(r"[a-f0-9]{64}", calibration[name]):
            raise ModelPackageError("Calibration input checksums are required")
    versions = manifest.get("libraries")
    if not isinstance(versions, dict) or set(versions) != {"torch", "numpy", "rdkit"} or any(
        not isinstance(value, str) or not value.strip() for value in versions.values()
    ):
        raise ModelPackageError("Inference library versions are required")
    thresholds = _read_json(root / "thresholds.json")
    validate_thresholds(thresholds)
    return manifest, thresholds


def publish_package(checkpoint, destination, *, thresholds, model_version, export_revision,
                    calibration, libraries):
    """Publish to a new directory only. An existing package is never replaced."""
    target = Path(destination)
    if target.exists():
        raise ModelPackageError("Destination already exists; choose a new package directory")
    validate_thresholds(thresholds)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-package-", dir=target.parent))
    try:
        shutil.copyfile(checkpoint, staging / "weights.pt")
        (staging / "thresholds.json").write_text(json.dumps(thresholds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {"schema_version": 1, "model_version": model_version,
                    "export_revision": export_revision, "model": MODEL_SPEC,
                    "fingerprint": FINGERPRINT_SPEC, "labels": LABELS, "top_labels": TOP4_LABELS,
                    "calibration": calibration, "libraries": libraries,
                    "files": {name: checksum(staging / name) for name in ("weights.pt", "thresholds.json")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        validate_package(staging)
        # Same-filesystem rename: consumers cannot see a half-written directory.
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target
