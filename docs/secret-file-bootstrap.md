# Opt-in credential-file bootstrap

8 November 2026. Work continues on operational branch
`feature/infra-vm-nats-client-auth`, as explicitly requested by the owner.
Future related work starts here; `develop` stays behind until owner integration.
No merge, secret provisioning or VM operation is implicit.

## Activation and format

The default Docker CMD remains `python -m main`. The opt-in alternative is:

```bash
python -m bootstrap_secrets
```

Set `PY_ENV=staging` or `production` outside the file, and
`MERCURION_BOOTSTRAP_FILE` to its absolute path. The future runtime manifest
chooses this command only after the operational gate. No current Compose is
changed. Both modules enter the Docker source allowlist and final runtime
inventory, without new dependencies or training components.

| JSON field | Contract |
| --- | --- |
| `schemaVersion` | Integer `1` |
| `environment` | `staging` or `production`, exactly matching external `PY_ENV` |
| `variables` | Exactly `NATS_USERNAME` and `NATS_PASSWORD`, both present |

No extra top-level fields or duplicate JSON keys. Strict UTF-8 with optional BOM,
maximum 1 MiB, no NaN/Infinity. Values must be nonblank strings without NUL or
isolated Unicode surrogates. Values are literal: no dotenv interpolation, shell,
evaluation or URL encoding. The bundle owns the complete credential pair;
pre-existing values are rejected even when identical. All entries are validated
before environment injection. Canonical Pydantic validation still runs afterwards;
the file does not supply other required app settings.

This allowlist is narrower than Nest's: no Python control variables, environment
selector, JWT material, service auth overrides or broker URL. The public JWT PEM
stays a separate read-only file selected by `JWT_PUBLIC_KEY_PATH`. No private
signing key belongs in Tox21.

## Filesystem and errors

Only regular files, no directories or symlinks. POSIX opens use no-follow and
non-blocking flags, then check the descriptor again. Group-write/execute and
all others' permissions are rejected; group-read is allowed. The file must be
readable by the actual UID 10021; Windows ACL equivalence is not claimed.
Reads are bounded to 1 MiB plus one byte, including growth after stat; descriptors
close on success and failure. The consumer never deletes or rewrites the file.

Only the current Python process receives values. No credential is added to argv,
Docker `Config.Env` or an env file by this adapter. Root/Docker administrators
and process memory remain trusted; this is not forensic erasure.

Adapter failures expose `INVALID_SECRET_BOOTSTRAP` plus a fixed reason and exit
1. Other startup exceptions expose only `APPLICATION_START_FAILED` /
`application-startup`. No raw exception, path, arbitrary field name or supplied
value is printed by the launcher. Imports of `main` and scientific modules are
deferred until file validation/injection. Missing input fails rather than
falling back to the normal command. Internal app diagnostics retain their own
behavior and are not claimed fully audited here.

## Evidence and limits

- Application suite: **122 tests on Linux, all passed**; **122 on Windows,
  120 passed and two POSIX tests skipped**. Parser, atomicity, duplicate/collision,
  UTF-8, size/growth, permissions, symlink replacement race, descriptor cleanup
  and launcher error/order checks.
- Actual CLI subprocess tests use a sentinel app module: missing file prevents
  its import, valid file injects before start. This is a launcher test double.
- Real isolated NATS smoke reads eight synthetic credential files before actual
  worker connections: 10 authenticated RPCs, 5 legacy RPCs and 8 explicit worker
  auth denials. Inference/PubChem are synthetic; RDKit is real. This composes file
  loading with `main.run`, independently of the CLI sentinel.
- Linux checks use the existing dependency image, non-root, with selected current
  source mounted read-only. No local fresh release-image build is claimed. CI
  includes tests, updated runtime inventory and broker smoke against its newly
  built image; require its exact-SHA result separately.

The [broker-auth smoke command](nats-client-auth.md) is unchanged; its
`--source-overlay` mode mounts both bootstrap modules too. No dependency install,
operational secret/PEM/model read, provider call, SSH or VM mutation was needed.
Receivers, protected transport, mount ownership and restart/recreate lifecycle
remain separate operational work.
