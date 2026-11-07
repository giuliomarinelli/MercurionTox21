# T21-02 — Runtime contract

6 November 2026. Implemented on `feature/T21-02`, based on the T21-01 documentation branch.
No merge, deployment, real key rotation or external service startup is part of this task.

## Scope decision

Mercurion is a portfolio case study maintained by one developer. This task demonstrates
explicit configuration, preserved wire contracts, tested startup/cleanup and an honest
authentication boundary. It does not introduce a multi-key registry, hot key reload,
Base64 key injection, a secret provider or a production operations platform.

Authentication remains all-or-nothing through `GLOBAL_SKIP_AUTH_FLAG` in `main.py`:

- `True` is the unchanged default: validate request shape, skip JWT verification and do
  not read any key file. This is public internal RPC, not a claim of authenticated RPC.
- `False` requires a readable **RSA public key in a `.pem` file**, including staging and
  production. Read and validate it once during bootstrap, before loading the model or
  connecting to NATS. An unreadable, invalid, private or non-RSA key aborts startup.
- `JWT_PUBLIC_KEY_PATH` selects the file; it is not the key content. No Base64 alternative
  or environment variable switches the auth flag. Deployment can supply a read-only file
  mount without putting its material in the application image.
- Preserve RS256 and audience `mercurion-api`. A single public-key snapshot verifies
  requests until process restart. No coordinated rotation, revocation registry or `kid`
  selection is implemented here. Replacing Web's signing key while this authenticated
  worker still uses the old public key can break validation; merely adding a `kid` header
  with the same signing key is not itself a key change.

Web's existing opt-in rotation code and its historical tests remain intact. Adopting
coordinated rotation is deferred for the current portfolio scope. Enabling auth for RPC
that Web currently calls without a token is also a separate consumer decision. This task
does not enable it or reinterpret Web login/session validation as worker authentication.

## Configuration

`get_config` in [config.py](../config.py) takes a snapshot of process variables; it does
not mutate `os.environ`. An explicit mapping and root directory can be supplied by tests.

| Input | Contract |
| --- | --- |
| `PY_ENV` | Exact lowercase `development`, `test`, `staging`, `production`; default `development`. Blank/unknown names fail before any dotenv lookup |
| `NATS_URL` | Required nonblank string with `nats://` or `tls://`, host and valid optional port; no resource path/query/fragment. Credentials, if supplied, are not included in configuration errors or startup logs |
| `VERSION` | Required nonblank string; `local-dev` is valid for local use, not an immutable release identity |
| `JWT_PUBLIC_KEY_PATH` | Nonblank file path, default `public.<environment>.pem`, or `public.pem` in production. File read only when auth is enabled |

Development/test optionally read `.env.development` / `.env.test` from the working root;
process variables take precedence. Dotenv interpolation is disabled: values are literal,
not templates using another variable. `PY_ENV` is selected from the process before reading
the file. Staging/production **never read dotenv files**, including `.env.staging` and
`.env`: supply process variables explicitly. This is an intentional configuration change
from the previous loader, and existing deployment definitions must be aligned in T21-06.
The old Compose does not already supply a complete configuration.

Invalid input reports field names, not supplied values or full Pydantic errors. The CLI
reports startup failure without echoing NATS credentials or key material. Main returns
nonzero on configuration, model-load or terminal connection failure.

## Bootstrap and lifecycle

1. Import defines functions; it does not load configuration, PEM, thresholds or weights,
   set Torch threads, open NATS or start background work. Native RDKit imports still exist;
   this task does not remove scientific dependencies from the environment.
2. Explicit `run` loads validated configuration and, only when needed, the public PEM.
3. `load_inference_runtime` loads Torch, the existing threshold files and weights, sets
   the existing thread limits, loads `state_dict` with `weights_only=True`, selects device
   and calls `eval`. Thresholds are bound to the predictor used by handlers. No training.
4. Connect NATS, register the five existing subjects and flush with a five-second timeout.
   Only then mark the process-local readiness event and print `Ready`.
5. Disconnection clears readiness; the client's reconnect callback restores it only after
   startup subscriptions were completed and while stop has not been requested. The existing
   `nats-py` reconnect defaults remain: 60 attempts, two-second wait per server in the pinned
   client. This is not a new retry supervisor. Terminal client close ends the worker with
   failure instead of leaving its main wait alive indefinitely.
6. Stop/cancellation or partial startup failure clears readiness. Drain a connected client
   with a five-second bound, then close even if drain fails; close has its own five-second
   bound. No guarantee that an already running external PubChem thread is forcibly stopped.

The CLI installs SIGINT/SIGTERM event handlers where supported by the event loop. Windows
uses `asyncio.run`'s KeyboardInterrupt cancellation path when those handlers are unsupported.
Tests use explicit events and fake clients; OS signal delivery and real broker recovery
remain runtime/container acceptance work. The readiness event is not a new HTTP endpoint
or a completed deployment probe.

## Compatibility and limits

The five request callbacks, JWT verifier and wire helpers retain the same AST as the
pre-task source; namespace construction now goes through a validated environment helper.
All five subject names and payload/result shapes remain unchanged. In particular, the
inference result has no extra `data` wrapper. Chemical-error category differences between
operations are preserved rather than silently normalized during a lifecycle change.

The export contract associating weights and thresholds, package checksums, corrupt-package
checks beyond ordinary load failures, real-model inference regression and runtime-only
Docker packaging remain T21-03/09/04. No model or threshold file was edited or claimed
validated by tests that replace the model loader.

`api.inference.predict` still accepts its previous three positional arguments; direct
callers load thresholds explicitly on invocation. The service binds thresholds once during
bootstrap, so request handling does not reread them. Importing inference no longer opens
threshold files. No coherence between the currently tracked weights and thresholds is
inferred from successful imports or handler tests.

## Verification

Focused baseline before source changes: **25 tests passed** from the existing handler and
RDKit groups. Final focused run with the existing virtualenv:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_config tests.test_runtime tests.test_nats_requests tests.test_rdkit_api
```

**54 tests passed**: 8 configuration, 21 PEM/import/lifecycle and the 25 existing handler/
RDKit tests. RSA tests generate synthetic keys in memory and temporary files: valid and
invalid signatures, expiry, audience, algorithm, rejected PEMs and authenticated staging/
production bootstrap. Lifecycle tests use fake NATS/model dependencies; no real broker,
user key, token, dotenv configuration or external API is used.

The initial attempt to include `tests.test_training_pipeline` was interrupted while
importing pandas native modules. A diagnostic rerun identified that import stack and was
also stopped; no test result was obtained for that group. It is not counted as passed or
repaired. Training/dependency remediation remains outside T21-02.

A separate isolated smoke check on source commit `884e74fccd78dd8bb7cdcf8847b29c06551d709d`
imported `main`, loaded the actual tracked weights/thresholds through
`load_inference_runtime` and predicted `CCO` on CPU. It returned exactly the four expected
labels with probabilities in `[0, 1]`, without connecting NATS or calling PubChem. No model
files changed. This verifies the changed loader's native-library import order and one real
prediction; it does not prove checkpoint/threshold provenance, accuracy or complete
prediction regression coverage.

No fresh install, container build, real NATS reconnection, browser validation or remote
application CI was performed. Dedicated Tox21 CI is planned in T21-05. Source/diff checks,
local unit tests and this isolated real-weight smoke check are not full deployment acceptance.
