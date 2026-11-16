# Application containers

The single root `Dockerfile` provides Linux amd64 CPU targets `test`, `staging`
and `production` (default). All start `python -m main` as UID/GID `10021:10021`.
`PY_ENV` selects the corresponding NATS namespace and configuration policy;
runtime credentials and broker addresses are not build inputs.

An [opt-in credential-file launcher](secret-file-bootstrap.md) is included as
`python -m bootstrap_secrets`. Default CMD and all existing target behavior are
preserved; the alternative is selected explicitly by a future runtime manifest.

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

The digest-pinned Wolfi/glibc base, Python 3.11.18 and hash-locked Linux CPU dependency profiles
are shared. `test` additionally requires the validation stage to pass the
159 self-contained unittest cases. Training dependencies and synthetic tests
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
broker and worker, probes all five subjects using the default CMD, then checks
broker outage/recovery without worker restart and graceful SIGTERM exit.
PubChem is tested through request validation, without provider access.
No host ports or existing services are used;
owned resources are removed in `finally`. It needs the pinned broker image
available locally or Docker network access to pull it. These commands are
local evidence; [T21-05 CI](ci.md) runs these checks remotely and retains diagnostics.

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

## Local evidence — 6 November 2026

All three targets were built from Git archive source
`a3a408ae34fd380d13fa4d8d866bc8b961992ffd`, version `t21-04-qa`, with the
previously exported package `qa-6f19c86` (627 validation samples). The package
is QA evidence, not an assigned release or a claim about scientific quality.
The 99-case build gate passed. Image audit and isolated real NATS RPC/graceful
stop passed for every target. The production image additionally passed the
existing four-molecule/sixteen-prediction reference test offline, with its
test files mounted read-only from outside the image.

Synthetic context sentinels in Git/virtualenv/data/output/notebook/config/key
paths stayed excluded (53 admitted files, 156754 bytes). Builds failed for a
modified thresholds checksum, missing version and abbreviated source revision.
Compose was resolved with an explicit empty env file and synthetic values.
No user services, keys, dataset or original outputs were changed. QA-owned
broker/worker containers and networks were removed. Images/build cache and
logs remain local under `C:/Users/giuli/.codex/artifacts/t21-04`.

These are local Linux amd64 CPU results. Remote CI is T21-05, Web deployment
alignment T21-06, coordinated delivery T21-07 and integrated acceptance T21-08.
No image was published or deployed; no branches were merged.
