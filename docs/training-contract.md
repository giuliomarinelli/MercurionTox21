# T21-10 — Local training correctness contract

7 November 2026. This task checks pipeline correctness on tiny synthetic data;
it does not retrain the historical model or certify its scientific accuracy.

## Invariants fixed before implementation

- Keep the existing seeded, two-stage random row split and default 70/10/20
  proportions. Each retained row belongs to exactly one split, features stay
  aligned with its twelve labels in `tox21_labels` order, and the same input
  and seed reproduce the arrays. Duplicate molecules or scaffolds can still
  cross splits: changing that methodology needs a separate decision.
- Reject missing label columns, nonbinary/missing labels, empty usable inputs,
  and invalid split fractions before writing arrays. Do not impute unknown
  assay results as negative. Preserve omission of unparsable SMILES; missing
  or non-string SMILES are omitted by the same preprocessing boundary. Validate
  labels on retained rows. A fully missing/ambiguous dataset is not usable.
- Training loaders accept only nonempty binary 2048-bit features and twelve
  binary targets per row. Load with pickle disabled. Invalid arrays fail before
  optimization, rather than propagating NaNs into loss or checkpoints.
- Calibration accepts finite probabilities in [0,1] aligned with nonempty
  binary targets. Preserve the existing 0.10..0.85 grid (step 0.05), strict
  `probability > threshold`, macro F1, first strict improvement and fallback
  0.5 when every candidate scores zero. Constant labels remain supported with
  sklearn's existing F1 behavior; their threshold is not evidence of quality.
  ROC-AUC is undefined without both classes; no artificial AUC is introduced.
- Early stopping saves only an improvement strictly greater than delta,
  resets its counter on improvement and stops at patience consecutive misses.
  Loss must be finite and patience/delta valid; invalid loss cannot replace a
  good checkpoint. The selected checkpoint, not the final epoch, feeds final
  calibration and export. Export must not read the held-out test arrays.
- No change to architecture, fingerprint, label order, production auth, model
  package format, tracked datasets/weights/thresholds or dependency locks.
  Training and its tests remain absent from final inference images.

## Scientific evaluation remains separate

Before publishing a newly trained model, record seed/configuration, input/split
identity, dependencies, selected epoch/checkpoint and export identity. Report
per-label and aggregate held-out metrics, class counts and undefined AUCs.
Choose comparison tolerances and acceptance criteria before evaluating; do not
select thresholds on the test set. Historical provenance cannot be invented.
No full-data training or new-model quality report is part of this task.

## Validation

Use the existing locked Linux CPU training environment, offline:

```bash
python -B scripts/run_tests.py training --report-json training.json
python -B scripts/run_tests.py all --reverse --report-json all-reverse.json
```

All synthetic files/checkpoints/packages use temporary directories. Tests with
controlled fingerprint stubs check row identity/alignment, while existing real
RDKit fingerprint checks cover the scientific transformation separately.

### Local evidence

- Baseline `a99754c9e61051036153d3676b7ef5d568b0c1bc`, exact Tox21 CI
  [42360144406](https://github.com/giuliomarinelli/MercurionTox21/actions/runs/42360144406):
  application/containers and Required gate succeeded before implementation.
- Before guards, the initial 14 new test methods reproduced 22 failing subtests
  and five errors, including invalid input acceptance and nonfinite checkpoint
  replacement. Existing valid-data behavior tests also ran in that red phase.
- After guards: **21 training tests passed**; **123 self-contained tests passed**
  in normal order and **123 in reverse order**, including the real synthetic
  selected-checkpoint/calibration/export path. Counts overlap.
- Used the existing locked Linux CPU image
  `mercurion-t21-baseline-validation` (`653a20e21a05`), network disabled, with
  candidate source/test mounts read-only. No dependency installation, full-data
  training, service restart, model refresh or tracked artifact mutation.
- Local reports/logs: `C:/Users/giuli/.codex/artifacts/t21-10/` (`before.log`,
  `training.json`, `all.json`, `all-reverse.json`, `all.log`). These are local
  evidence paths, not prerequisites for cloning or running the suite.

The feature-SHA CI supplies clean-build evidence separately; local success is
not a substitute for that gate. New-model scientific evaluation remains pending.
