# T21-05 — Application CI

**10 November policy update:** [strict zero-vulnerability enforcement](strict-vulnerability-policy.md)
now runs stdlib policy tests and a complete legacy/runtime/training/build source
scan before builds. Any severity, incomplete scan or scanner error fails the
verification job and Required gate. The image exporter uses the same rule.
Historical successes described below retain their original threshold; current
known findings must be fixed before CI can pass under policy v2.

The [file-bootstrap update](secret-file-bootstrap.md) adds its unit/CLI checks
to the application group (122 cases; POSIX checks run on Linux), updates the
runtime file inventory and supplies synthetic credentials from files in the
existing broker smoke. It preserves default CMD and dependency locks.

**8 November update:** [broker user/password smoke](nats-client-auth.md) is added
after mixed JWT service-auth checks, against the freshly built production image.
It verifies 10 authenticated RPCs, 5 legacy RPCs and 8 explicit auth denials, with
synthetic inference/PubChem and no real credentials. Local application coverage
is now 107 tests; counts below retain historical T21-05 evidence. The updated
workflow itself still requires an exact-SHA remote run.

The [Tox21 CI workflow](../.github/workflows/ci.yml) runs on development/feature/
release/hotfix/bugfix pushes, PRs targeting develop/master and manual dispatch.
It never publishes images, deploys, merges branches or needs personal secrets.
Only read access to repository contents is granted; actions are pinned by SHA.

The Linux amd64 CPU verification job uses the pinned Python 3.11.18 Docker
base and hash-locked profiles. A fresh hosted runner performs clean dependency
installation (no persistent dependency or image cache). Steps fail on nonzero
exit; piped commands retain `pipefail`. The stable **Required gate** runs even
when verification fails/is skipped, and succeeds only for verification success.
This publishes a check; configuring branch protection remains an operator choice.

Evidence is separated into:

1. AST parsing of tracked Python files without import side effects.
2. Validation image build, locked install, `pip check`, and 107 self-contained tests.
3. Explicit offline unittest groups: application 93, model/export 9, training 5.
   These overlap the build suite and are not 198 distinct cases. JSON reports
   contain counts, status and failing test diagnostics, alongside text logs.
4. QA package export using the selected tracked checkpoint plus **only**
   `X_val.npy`/`y_val.npy` mounted read-only. No train/test dataset, retraining,
   external model download or mutation of original outputs. The package records
   exact source SHA and calibration hashes; old training provenance is not implied.
5. Three final target builds, metadata/content audit and offline repeatable inference.
6. Four-molecule/sixteen-prediction numerical reference using the real package.
7. Five-subject NATS smoke on each target, broker outage/recovery without worker
   restart, graceful SIGTERM exit 0. PubChem gets an invalid request that is
   rejected before the provider boundary; real PubChem success is not claimed.

Smoke networks are internal, brokers pinned, no host ports published. The worker
uses the normal CMD, non-root, read-only filesystem and temporary `/tmp`. RPC
responses establish readiness, not logs/process existence. The probe connects
to the owned broker's IP to bound the outage check without waiting for Docker
DNS on a stopped container; the application still uses its ordinary NATS alias.
Workers/brokers/networks are removed in `finally`; diagnostics are saved first.

Diagnostics are uploaded on success or failure, with a 14-day retention, under
`tox21-reports-<sha>-<run_attempt>`: build/test logs, JSON results, smoke
worker/broker logs and image identities. The successfully exported QA package
is uploaded independently as `tox21-model-<sha>-<run_attempt>`, also when later
verification fails. Test/export containers use the runner's UID/GID for writable
bind mounts; a package access failure cannot prevent diagnostics upload. Upload failure
fails verification too. Hard runner termination may prevent cleanup/upload;
the hosted runner's isolated resources then expire with it. No registry release
identity or coordinated Web delivery is established by a green CI run.

Useful local commands (Docker and matching source required):

```bash
docker build --target validation --tag tox21-validation .
docker run --rm --network none tox21-validation /opt/venv/bin/python -B scripts/run_tests.py application
python -B scripts/smoke_container.py <built-image> --target test --report-dir <external-results-directory>
```

Container inputs/build arguments are documented in [containers.md](containers.md).
Use an external report directory locally; CI's `artifacts/` is runner-local.
TLS/JWT-active tests, provider success, scientific quality and integrated Web
acceptance remain separate work. T21-06 next aligns the old Web deployment refs.

T21-07 adds a final archive/SBOM/security gate: [contract and current block](delivery.md).
