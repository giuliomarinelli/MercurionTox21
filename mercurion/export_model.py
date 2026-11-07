"""Local export: load one immutable checkpoint snapshot, calibrate on validation, publish."""
import argparse
import hashlib
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import tempfile

# Torch precedes sklearn's native dependencies on Windows.
import torch
import numpy as np

from mercurion.calibration import find_best_threshold, find_per_label_thresholds
from mercurion.model import MercurionMLP
from mercurion.model_package import ModelPackageError, publish_package
from mercurion.model_spec import LABELS, MODEL_SPEC


def validation_predictions(model, features, targets, *, device="cpu"):
    if (features.ndim != 2 or features.shape[1] != MODEL_SPEC["input_dim"] or features.shape[0] == 0
            or targets.shape != (features.shape[0], len(LABELS))
            or not np.isin(features, [0, 1]).all() or not np.isin(targets, [0, 1]).all()):
        raise ModelPackageError("Expected nonempty binary validation features/targets with compatible dimensions")
    model.eval()
    results = []
    with torch.no_grad():
        for offset in range(0, len(features), 128):
            batch = torch.as_tensor(features[offset:offset + 128], dtype=torch.float32, device=device)
            results.append(torch.sigmoid(model(batch)).cpu().numpy())
    probabilities = np.concatenate(results)
    if probabilities.shape != targets.shape or not np.isfinite(probabilities).all():
        raise ModelPackageError("Invalid validation predictions")
    return probabilities


def export_model(checkpoint, validation_root, destination, *, model_version, export_revision):
    if Path(destination).exists():
        raise ModelPackageError("Destination already exists")
    root = Path(validation_root)
    # Snapshot the exact inputs used by calibration; never load X_test/y_test.
    features_bytes = (root / "X_val.npy").read_bytes()
    targets_bytes = (root / "y_val.npy").read_bytes()
    features = np.load(BytesIO(features_bytes), allow_pickle=False)
    targets = np.load(BytesIO(targets_bytes), allow_pickle=False)
    checkpoint_bytes = Path(checkpoint).read_bytes()
    with tempfile.TemporaryDirectory(prefix="mercurion-export-") as directory:
        snapshot = Path(directory) / "weights.pt"
        snapshot.write_bytes(checkpoint_bytes)
        model = MercurionMLP()
        model.load_state_dict(torch.load(snapshot, map_location="cpu", weights_only=True))
        probabilities = validation_predictions(model, features, targets)
        global_threshold, _ = find_best_threshold(targets, probabilities)
        thresholds = {"global": global_threshold,
                      "per_label": find_per_label_thresholds(targets, probabilities)}
        return publish_package(
            snapshot, destination, thresholds=thresholds, model_version=model_version,
            export_revision=export_revision,
            calibration={"split": "validation", "samples": len(features),
                         "features_sha256": hashlib.sha256(features_bytes).hexdigest(),
                         "targets_sha256": hashlib.sha256(targets_bytes).hexdigest()},
            libraries={name: version(name) for name in ("torch", "numpy", "rdkit")},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--validation-root", default="data/processed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--export-revision", required=True,
                        help="Full source SHA of the export implementation, not a claim about training provenance")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = export_model(args.checkpoint, args.validation_root, args.output,
                        model_version=args.model_version, export_revision=args.export_revision)
    print(f"Exported validated model package: {path}")


if __name__ == "__main__":
    main()
