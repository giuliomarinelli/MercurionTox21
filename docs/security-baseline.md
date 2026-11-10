# Security baseline — 7 November 2026

Historical critical-only evidence. From 10 November the owner's [strict policy](strict-vulnerability-policy.md)
blocks every severity; the results below are not eligibility under policy v2.

Focused follow-up to the T21-07 critical-vulnerability block. The branch
`feature/T21-BASELINE` starts from preserved `feature/T21-07`, so it also contains
the image identity, archive evidence and fail-closed security gate. No merge is
performed. No exception or severity threshold is changed.

## Deliberate changes

- Python **3.10.11 → 3.11.18**, base Debian **11.7 → 13.7**. Selected image:
  `python:3.11.18-slim-trixie@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79`.
- Pillow **12.3.0 → 13.0.0**, an explicit runtime security pin. Pillow 13 needs
  Python >=3.11: [upstream package metadata](https://pypi.org/pypi/pillow/13.0.0/json).
- CPython 3.11 is the smallest tested step that preserves the pinned scientific
  stack; no unsupported claim is made about newer Python/native-wheel combinations.
  [Python branch lifecycle](https://devguide.python.org/versions/).
- Runtime/training hash locks regenerated from cp311-compatible distributions:
  still **23/39 packages**. Only Pillow's library version changes. Other native
  wheel hashes change because the interpreter ABI changes. Torch 2.13.0+cpu,
  NumPy 1.24.4, RDKit 2024.3.5 and all other resolved versions stay pinned.
- Pip-removal paths now refer to Python 3.11. No apt upgrade or floating package
  installation is added; the base digest and full hash locks remain explicit.

The root research umbrella now pins Pillow 13 as well, so it also requires a
Python >=3.11 environment. Only the Linux amd64 CPU application/training-export
profiles are verified; optional DeepChem research dependencies are not certified.
The existing Windows virtualenv is untouched, not silently upgraded.

## Regression boundaries and local evidence

No original weights, thresholds, dataset or model package is changed. No real
training is run. All 107 self-contained tests pass on the clean Linux profiles.
The separate original model-reference test passes with four molecules/sixteen
predictions at the existing `1e-6` probability tolerance and exact thresholds/
booleans. The old exported package remains compatible because its recorded
Torch/NumPy/RDKit versions are unchanged.

Pillow PNG roundtrip and Matplotlib Agg PNG generation pass. An extra RDKit Draw
probe requires `libXrender.so.1`, absent in both old and new minimal images. This
is an existing optional-drawing limitation, not a regression or a reason to add
an unrelated OS graphics stack; the five active APIs do not use RDKit Draw.

The new bare Python base scan reports **zero CRITICAL findings**, with other
severities retained. Application eligibility still requires scanning the actual
final image, not extrapolating this base result. The existing CI builds/audits all
targets, checks the model reference, smoke-tests NATS/outage/stop, then generates
SBOM, verifies archive save/load and applies the unchanged critical gate. Exact
source SHA, reports and DB date identify each result; no permanent vulnerability
guarantee follows from one successful scan.

Auth code and global policy are unchanged. The proposed per-service JWT overrides
are a future documentation-only task in the Web roadmap. Coordinated Web image
records and T21-08 integrated acceptance remain separate after this baseline.
