# Application containers

The single root `Dockerfile` provides Linux amd64 CPU targets `test`, `staging`
and `production` (default). All start `python -m main` as UID/GID `10021:10021`.
`PY_ENV` selects the corresponding NATS namespace and configuration policy;
runtime credentials and broker addresses are not build inputs.

## Build

Use a clean source checkout and a separately exported, validated
[model package](model-package.md). The package directory must contain only
`weights.pt`, `thresholds.json` and `manifest.json`. It is an explicit BuildKit
named context, not the repository's `outputs` directory. Never use a directory
containing personal files or keys as that context.

```powershell
$revision = git rev-parse HEAD
docker build --target test --build-context model_package=C:/artifacts/exported-model --build-arg BUILD_VERSION=local-qa --build-arg BUILD_REVISION=$revision --tag mercurion-tox21:test .
```

Repeat with `--target staging` or `--target production` and distinct tags.
Version must be nonblank and revision a complete 40-character Git SHA; labels
record both and the source repository. These arguments do not establish a
release or prove that a dirty checkout matches the supplied revision.

The pinned Python 3.10.11 base and hash-locked Linux CPU dependency profiles
are shared. `test` additionally requires the validation stage to pass the
99 self-contained unittest cases. Training dependencies and synthetic tests
exist only in that stage. All final images contain the same runtime code and
exported package; the test target adds only a small test-completion marker.
Every target validates and loads its package and runs a real CPU prediction
during build. An invalid explicit package fails; no legacy fallback is copied.

`.dockerignore` uses a positive source inventory. Git, host virtualenvs,
dotenv/PEM/key files, datasets, notebooks and `outputs` do not enter the main
context. Final layers copy explicit application modules and runtime dependencies;
training/export modules, pandas, scikit-learn, matplotlib, tests and pip are
absent. Torch remains necessary for inference. No compiler is added.

## Validate locally

```powershell
python -B scripts/check_container_image.py mercurion-tox21:test --target test --version local-qa --revision $revision
python -B scripts/smoke_container.py mercurion-tox21:test --target test
```

Repeat for staging/production. The first command checks identity/content and
repeatable real inference offline, non-root, with a read-only filesystem.
The second creates an isolated internal Docker network, its own pinned NATS
broker and worker, sends a real inference RPC using the default CMD, then
checks graceful SIGTERM exit. No host ports or existing services are used;
owned resources are removed in `finally`. It needs the pinned broker image
available locally or Docker network access to pull it. These commands are
local evidence; T21-05 adds remote CI and broader broker contract coverage.

For service execution use `--read-only --cap-drop ALL --security-opt
no-new-privileges:true --tmpfs /tmp:rw,noexec,nosuid,size=64m` and supply
`NATS_URL` on a network shared with the chosen broker. `/tmp` is the only
declared writable area. The baked `VERSION` can be overridden deliberately;
keep runtime/image identities consistent. No HTTP port/health check is invented
for this NATS worker. Docker stop should allow 15 seconds for drain/close.

The existing auth policy remains all-or-nothing and public internally by
default. If explicitly enabled in source, mount an RSA public PEM read-only
and set `JWT_PUBLIC_KEY_PATH`; do not bake keys or Base64 runtime material.
Secret rotation remains outside this task.

The [local Compose example](../docker_mt21/docker-compose.yml) uses target
`test`, requires `MODEL_PACKAGE_CONTEXT`, `BUILD_VERSION`, `BUILD_REVISION`,
`NATS_URL` and an existing `spectralume_network`. It does not create the broker.
Web deployment references are migrated separately in T21-06; the old Web beta
Compose is not a working deployment contract for these new containers yet.
