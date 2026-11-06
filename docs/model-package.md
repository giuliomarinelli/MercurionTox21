# T21-03 — Coherent model export and dependency profiles

6 November 2026. Work on `feature/T21-03`, based on unmerged `feature/T21-02`; no automatic
merge, application deployment or training on the tracked dataset was performed.

## Export contract

Run calibration/export locally, using a selected checkpoint and **validation** inputs.
No training examples or final test split are required by the exporter. From the repository
root in the training/export environment:

```powershell
$exportRevision = git rev-parse HEAD
python -m mercurion.export_model --checkpoint outputs/models/best_model.pt `
  --validation-root data/processed --output outputs/packages/my-model-v1 `
  --model-version my-model-v1 --export-revision $exportRevision
```

`my-model-v1` is an example identifier chosen by the operator, not an assigned model
release. Choose a **new destination directory**. The command never overwrites an existing
package or the original checkpoint/threshold files. It:

1. Snapshots the selected checkpoint and `X_val.npy`/`y_val.npy` bytes.
2. Loads that snapshot into the existing MLP on CPU with `weights_only=True` and `eval`.
3. Validates nonempty binary feature/target arrays and their dimensions; rejects nonfinite
   predictions. Missing labels represented by NaN are not silently imputed.
4. Predicts validation batches without gradients and uses the existing F1 threshold search
   for global and per-label thresholds. It never reads `X_test.npy` or `y_test.npy`.
5. Writes and validates a staging directory, then renames it on the same filesystem so a
   consumer sees the complete package. Failure cleans staging instead of publishing a
   partial model. Existing valid packages remain untouched.

The package contains exactly:

| File | Purpose |
| --- | --- |
| `weights.pt` | Byte-identical snapshot of the selected checkpoint |
| `thresholds.json` | Global and four per-label thresholds computed with that snapshot |
| `manifest.json` | Schema version, model identifier, export source revision, architecture, fingerprint, ordered labels, library versions, calibration input hashes and file hashes |

`export_revision` identifies the implementation used for export. It does **not** establish
the unknown training provenance of an old checkpoint. Likewise, calibration hashes record
exact inputs but do not prove the validation split was originally independent of training.
Hashes detect drift/incomplete delivery; the manifest is not a signed trust authority or a
scientific quality certificate. Dataset changes and quality evaluation remain explicit.

## Use in the service

Set `MODEL_PACKAGE_DIR` to the package directory in the worker's process environment, e.g.:

```powershell
$env:MODEL_PACKAGE_DIR = 'outputs/packages/my-model-v1'
python -m main
```

The runtime checks schema, exact architecture/fingerprint/labels, threshold bounds,
validation metadata, file inventory/hashes and Torch/NumPy/RDKit version compatibility
before loading weights or connecting NATS. Torch's `+cpu` build suffix may differ while
the upstream version must match. Incompatible weights also fail during strict state load.
There is no fallback to legacy files when an explicit package fails validation.

Without `MODEL_PACKAGE_DIR`, the previous `outputs/models/best_model.pt` and two threshold
files remain available for compatibility. That path does not acquire coherent-export or
provenance guarantees merely because it still loads. The package mode is the path to use
for the runtime-only application image; see [T21-04 containers](containers.md).

The verified runtime-only source snapshot and [dependency profiles](../requirements/README.md) exclude training,
calibration/export modules, pandas, scikit-learn and matplotlib. Training and inference
share only the fingerprint primitive, model definition/specification and label order.
The worker does not train, calibrate or download a model during startup.

## Training consistency

`train_model` now reloads the checkpoint selected by early stopping at the end of training,
re-evaluates the validation loader and writes both legacy threshold files from that model.
It no longer leaves per-label thresholds from an arbitrary last epoch. The legacy set of
separate files is not an atomic delivery format: publish a versioned package for delivery.
No existing weights, thresholds, datasets or historical model copies were changed here.

## Verification and remaining limits

Clean Linux x86_64/Python 3.10.11 CPU runtime and training/export installations were built
from the hash-locked closures with no index access during install, pinned setuptools and
`--no-build-isolation`; `pip check` and native imports passed. The runtime has 23 resolved
distributions and the training profile 39. Existing package versions were preserved;
Torch uses its official CPU build. [T21-04](containers.md) subsequently consolidated
the Dockerfiles; the installation evidence in this section remains the T21-03 result.

In the clean training environment, **74 tests passed**: package integrity/atomicity,
real-weight synthetic export, deterministic inference, shared fingerprints, selected
checkpoint recalibration and the existing application/training cases. The checkpoint
regression uses two tiny synthetic epochs in a temporary directory, not the tracked
dataset or output tree. The original four pipeline tests passed on this Linux target.

In the runtime-only environment, **63 tests passed** with training source and dependencies
absent. A separate offline check exported a copy of the tracked selected weights using a
copy of the validation arrays and predicted repeatedly through the runtime-only loader.
No NATS, PubChem, GPU or user configuration was involved. Preview QA metadata is not a
release identity; final audit export records the committed implementation revision.

The pre-existing Windows virtualenv still stalled in native scikit-learn imports during
the broader test attempt. That check was stopped and is not counted as passed or repaired.
Windows dependency remediation, GPU, full application/container startup and remote CI are
not certified by the Linux checks. Local audit files/logs live outside the repositories
under `C:/Users/giuli/.codex/artifacts/t21-03-*`.
