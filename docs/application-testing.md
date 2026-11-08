# T21-09 — Application regression suite

**8 November broker-auth update:** the application group now passes **107 tests**
on Windows and Linux. [NATS credential validation and isolated broker smoke](nats-client-auth.md)
extend configuration/lifecycle coverage; historical counts below describe the
original deliveries and must not be added to the current group.

T21-10 extends the training group: see the [training correctness contract and
current evidence](training-contract.md). The counts below describe the original
T21-09 delivery; current groups contain 21 training / 123 total tests.

6 November 2026. Work on `feature/T21-09`, based on unmerged `feature/T21-03`.
Authentication, model/data artifacts and the five request/result contracts are preserved.
No broker, PubChem provider, deployment or full training run is required for these groups.

## Explicit unittest groups

From this repository root, with the appropriate dependency profile installed:

```bash
python -B scripts/run_tests.py application
python -B scripts/run_tests.py model
python -B scripts/run_tests.py training
python -B scripts/run_tests.py all
```

| Group | Tests | Requirements and evidence |
| --- | ---: | --- |
| `application` (default) | 85 | Runtime dependencies only; configuration, RSA/PEM, handler logic, RDKit, PubChem boundary mocks, known-logit inference and package structure |
| `model` | 9 | Training/export dependencies; synthetic real weights, calibration/export, deterministic prediction and corrupt/incompatible package loading |
| `training` | 5 | Synthetic preprocessing/calibration and two tiny epochs with a selected earlier checkpoint; no tracked inputs or outputs |
| `all` | 99 | Above groups, without duplicate modules; excludes the explicit external-package reference group |
| `reference` | 1 | Runtime dependencies plus the matching real package; four molecules and sixteen label predictions checked against a pre-task CPU baseline |

The runner is a thin wrapper around standard `unittest`; it does not start services, install
dependencies or retrain on the real dataset. `--reverse` reverses the selected test cases
to check order independence. Missing dependencies remain errors, never implicit skips.
The opt-in reference module lives outside default `test_*.py` discovery; standard
`python -m unittest discover -s tests` remains a self-contained run on the training profile.

The runtime-only group passed with training source and pandas/scikit-learn/matplotlib
absent. Synthetic model weights used by the other groups live in temporary directories.
Use the [Linux CPU profiles](../requirements/README.md); the pre-existing Windows scientific
import problem remains outside this task. Application and reference checks passed in the
existing Windows environment too, without modifying it.

## Real-package reference

This group is intentionally separate from `all` and **fails** if the package is missing
or does not match the reference hashes:

```powershell
$env:MODEL_PACKAGE_DIR = 'path/to/the/matching/package'
python -B scripts/run_tests.py reference
```

For the local audit the package is
`C:/Users/giuli/.codex/artifacts/t21-03-packages/verified-6f19c86/`. It is not a repository
prerequisite or an automatically deployed model. A matching package can be reconstructed
using T21-03's documented validation export with the same weights and validation inputs.

[`model-reference.json`](../tests/fixtures/model-reference.json) records parent implementation
`6f19c869a0d99209fc78eb6a909d1f45897118a8`, weight/threshold SHA-256 values and a `1e-6`
absolute probability tolerance. Predictions were generated with that **pre-task** source,
verified against its committed files, not refreshed from the candidate implementation.
The new code matches four input molecules, all four labels, exact thresholds and positivity
flags on CPU. The loader already checks package metadata/library compatibility.

Do not regenerate the fixture automatically when a regression fails. First distinguish
code drift, intentionally changed model/thresholds, or platform/library differences, and
review the relevant update. This is prediction regression for one known package, not a
scientific accuracy or provenance certificate. New-model evaluation remains T21-10.

## Shared Python/Web wire fixture

[`tox21-wire.json`](../tests/fixtures/tox21-wire.json) is synthetic contract data. Python
handler tests verify direct and Nest-enveloped requests and all five success response
shapes. Scientific operations in that handler test are replaced; it does not claim a real
model prediction or a real PubChem response.

MercurionWeb vendors an identical fixture in
`MercurionWebNode/src/contracts/fixtures/tox21-wire.json`. Its targeted Jest test checks
requests/responses and the four environment namespaces against `NATS_CONTRACT_REGISTRY`.
Ordinary Web CI therefore does not need a sibling checkout. Update both fixtures deliberately
and verify equality when contracts change; wire fixtures are source test inputs, not runtime
configuration or a dynamic cross-repository dependency.

## Defects reproduced and corrected

1. `predict` accepted nonfinite logits, wrong output dimensions or invalid legacy threshold
   values. It now rejects them before producing an unusable prediction. Valid sigmoid,
   label mapping and strict `probability > threshold` behavior are unchanged. Known-logit
   tests check numeric values, equality at the threshold, no gradients, dtype/shape and
   absence of model/threshold mutation.
2. A failed NATS response was caught as an application exception and triggered a second
   attempted `InternalError` response. `_respond` now raises sanitized
   `ResponseDeliveryError`; callbacks propagate it without a second reply. Cancellation
   also propagates. JSON encoding is done before the transport boundary, with nonfinite
   numbers rejected: an unserializable result remains an application error that can send
   one `InternalError`, not a transport failure. Tests cover normal/error replies and all
   five callbacks.

Tests reproduced these failures before the fixes. No automatic retry policy, new error
wire shape or new authentication mode was introduced. Real broker delivery behavior still
belongs to T21-05/08.

One audit clarification is also protected: `are_same_structure` already catches invalid
SMILES/canonicalization failures and returns `False`; it does not need to be normalized
into the other operations' `Invalid SMILES` response. No product-policy change was made.

## Verification

- Clean Linux CPU training/export profile: **99 tests passed**, also **99 in reverse order**.
- Clean Linux runtime-only profile: **85 application tests passed**, and the explicit
  real-package reference passed separately. All container test runs used `--network none`.
- Existing Windows virtualenv: **85 application tests** and the separate reference passed.
- Missing reference package produced exit code 1 with an explicit error, not a skip.
- Web: **47 targeted Jest tests** across the shared fixture, existing NATS registry and
  RDKit contract runtime suites passed; targeted ESLint passed.

Counts overlap and must not be added. Parent data, weights, thresholds and dependency locks
were not modified. No fresh dependency install, remote application CI or user-service
restart was performed here. The gap matrix and deferred broker/container/scientific
acceptance are maintained in MercurionWeb's T21-09 report.
