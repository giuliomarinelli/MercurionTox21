# MercurionTox21

MercurionTox21 is the Python inference and chemistry component of the Mercurion project,
maintained in its own [Git repository](https://github.com/giuliomarinelli/MercurionTox21).
The web application and its NATS consumers live in
[MercurionWeb](https://github.com/giuliomarinelli/MercurionWeb).

The application starts with `python -m main` from this repository root. It serves MLP
toxicity predictions, RDKit operations and PubChem name lookups over NATS. Model training
runs locally as a separate workflow. A checkout beside MercurionWeb, reachable there as
`../MercurionTox21`, is a local development convention, not part of the component's name
or a required layout for distributing its container.

## Repository map

| Path | Responsibility |
| --- | --- |
| [main.py](main.py) | Application bootstrap and five NATS request handlers |
| [config.py](config.py), [schemas/schemas.py](schemas/schemas.py) | Environment configuration and request validation |
| [api/inference.py](api/inference.py) | SMILES fingerprint and MLP prediction |
| [api/rdkit.py](api/rdkit.py), [api/pcp.py](api/pcp.py) | Chemistry operations and PubChem client |
| [mercurion/model.py](mercurion/model.py), [mercurion/labels.py](mercurion/labels.py) | Network definition and ordered output labels |
| [mercurion/train.py](mercurion/train.py), [mercurion/preprocessing.py](mercurion/preprocessing.py) | Local training and dataset preparation |
| `outputs/models/best_model.pt`, `outputs/best_threshold*.json` | Selected weights and inference thresholds |
| `data/`, `notebooks/` | Local model development inputs and exploration |
| `tests/` | Application regression tests, pipeline tests and evaluation scripts |

## Clone and install

```bash
git clone --branch develop https://github.com/giuliomarinelli/MercurionTox21.git
cd MercurionTox21
```

These instructions describe the application source on `develop`; they do not assume
that the repository's default branch contains the same runtime or test inventory.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The virtualenv is created locally and ignored by Git. Do not copy another computer's
virtualenv. The commands above retain the legacy development/research requirements.
T21-03 verified clean **Linux x86_64 / Python 3.10.11 / CPU** installs from separate
[runtime and training/export hash locks](requirements/README.md). Use the documented
profile procedure for that target; Windows/GPU compatibility is not implied by it.
The existing application Dockerfiles still await consolidation in T21-04.

## Configure and start the application

Start an appropriate non-production NATS broker first. Then, from this repository root
with the virtualenv activated, explicitly configure the worker. For the conventional Web
development broker on host port `4223`, Windows PowerShell uses:

```powershell
$env:PY_ENV = 'development'
$env:NATS_URL = 'nats://localhost:4223'
$env:VERSION = 'local-dev'
python -m main
```

Linux/macOS:

```bash
export PY_ENV=development
export NATS_URL=nats://localhost:4223
export VERSION=local-dev
python -m main
```

The broker address is configurable; inside a container, `localhost` refers to that
container, not to the host or a different service. Match `PY_ENV` to the Web consumer's
environment. `local-dev` is an explicit development version string, not a release identity.
Launching the application requires an existing compatible model package; it does not
require training again at every startup.

Without `MODEL_PACKAGE_DIR`, the legacy loader expects these files relative to the working directory:

- `outputs/models/best_model.pt`
- `outputs/best_threshold.json`
- `outputs/best_thresholds.json`

Weights alone are insufficient: the network definition, fingerprint parameters, ordered
labels and thresholds must match. The selected checkpoint and thresholds must be exported
coherently. [T21-03 export](docs/model-package.md) reloads a selected checkpoint, calibrates
on validation inputs and publishes a new checked package. Set `MODEL_PACKAGE_DIR` to use
it; an invalid explicit package never falls back to legacy files. Training also recalibrates
final thresholds from its selected checkpoint. None of this retroactively certifies or
changes the old model/threshold files currently checked in.

Configuration is defined by [config.py](config.py) and `Configuration` in
[schemas/schemas.py](schemas/schemas.py):

| Setting | Current behavior |
| --- | --- |
| `PY_ENV` | Exact lowercase `development`, `test`, `staging`, `production`; default `development`; other/blank values fail |
| `NATS_URL` | Required nonblank `nats://` or `tls://` broker URL |
| `VERSION` | Required nonblank string |
| `JWT_PUBLIC_KEY_PATH` | Optional override; otherwise `public.<PY_ENV>.pem`, or `public.pem` for production |
| `MODEL_PACKAGE_DIR` | Optional directory containing `weights.pt`, `thresholds.json`, `manifest.json`; validated before startup. Unset retains explicit legacy compatibility |

Development/test optionally read `.env.development` / `.env.test`; process variables take
precedence and dotenv values are literal, without interpolation. Select `PY_ENV` in the
process before startup. Staging/production never read dotenv files: supply process
variables explicitly. This changes the previous `.env.staging` / `.env` behavior; the
deployment definitions still require alignment. No personal configuration or key file is
supplied by the clone. See the [T21-02 runtime contract](docs/runtime-contract.md).

## NATS contracts and authentication

| Base subject | Operation / success response |
| --- | --- |
| `inference.tox21.smiles` | Four prediction labels, each with `probability`, `threshold`, `is_positive` |
| `rdkit_api.get_molecule_properties` | `data` containing molecular descriptors |
| `rdkit_api.to_canonical_smiles` | `data` containing canonical SMILES |
| `rdkit_api.are_same_structure` | `data` containing a boolean |
| `pcp_api.get_iupac_name_from_smiles` | `data` containing `iupac_name` |

Production uses the base subject. Other environments prefix it with `<PY_ENV>.`, for
example `development.inference.tox21.smiles`. Requests may be direct JSON objects or Nest
envelopes containing `data`. Errors use an `error` string. Inference returns the four
labels directly, without an extra `data` wrapper: `SR-ATAD5`, `NR-AhR`, `SR-MMP`, `SR-p53`.

**Current behavior:** `GLOBAL_SKIP_AUTH_FLAG=True` in `main.py` enables public internal
mode. Request payloads are still validated, but access tokens are ignored and a JWT key
is not required for this mode. This is separate from Web user authentication. The code
also has an authenticated path that verifies RS256 tokens with audience `mercurion-api`
using the configured public key. Auth remains all-or-nothing and there is no environment
variable that switches the global flag. When enabled, it requires a readable RSA public
`.pem` file also in staging/production; no Base64 key input, hot reload or multi-key rotation
is provided. Key failure stops startup, never enables public mode as a fallback. Adoption
of coordinated rotation is deferred for the portfolio scope; Web's existing opt-in code
is preserved. Do not interpret JWT code as proof that authentication is active.

## Local model training and evaluation

Run these commands explicitly from the repository root only when preparing a new model:

```bash
python -m mercurion.preprocessing
python -m mercurion.train
```

Preprocessing writes arrays under `data/processed/`. Training writes model/threshold files
under `outputs/`; it uses training loss and a separate validation split per epoch, with
F1/ROC-AUC metrics and early stopping. These commands can overwrite development outputs
and are not prerequisites for starting an already prepared inference package.

Post-training evaluation scripts use the local model and processed test dataset:

```bash
python -m tests.test_auc_score
python -m tests.test_roc_curves
```

The latter writes ROC plots. These are scientific evaluation tools, not assertion-based
`unittest` suites or substitutes for service regression tests. Threshold calibration must
use validation data, keeping the final test set separate.

## Application and pipeline tests

T21-09 provides [explicit standard-unittest groups and reference regression](docs/application-testing.md):

```bash
python -B scripts/run_tests.py application
python -B scripts/run_tests.py all
```

The default application group has 85 tests and needs only runtime dependencies. The
self-contained `all` group has 99 and requires the training/export profile. The separate
`reference` group checks a matching real package; it fails when that package is missing
instead of silently skipping. Linux checks also pass in reverse order; no retraining on
the real dataset or external services are part of these tests.

The existing explicit groups are:

```bash
python -m unittest tests.test_nats_requests
python -m unittest tests.test_rdkit_api
python -m unittest tests.test_training_pipeline
```

T21-02 adds explicit configuration and lifecycle/auth groups:

```bash
python -m unittest tests.test_config tests.test_runtime
```

The NATS handler tests replace broker, configuration and inference/model dependencies;
they do not connect to a real NATS server or prove prediction with real weights. RDKit
tests exercise the native chemistry library. Pipeline tests use synthetic data and replace
Torch/training dependencies where appropriate; they do not perform full training. The
groups still require their imported Python/native dependencies to be installed.

The original design inventory identified 18 handler methods, seven RDKit methods and four
pipeline methods. T21-02 verified 54 focused application tests, including the 25 handler/
RDKit cases, configuration, synthetic RSA signatures and fake-client lifecycle tests;
[results and limits](docs/runtime-contract.md). A separate smoke check loaded the tracked
weights and produced one CPU prediction without NATS/PubChem. The pipeline group was
interrupted during a pandas native import and has no new result. Broader real-model
regression, real NATS recovery and dedicated application CI remain separate acceptance work.

T21-03 subsequently passed **74 tests on clean Linux CPU** and **63 in a runtime-only
environment**, including coherent export and selected-checkpoint regression. Its four
original pipeline tests passed on Linux; the Windows scientific import issue remains
unresolved. [Package, commands and evidence](docs/model-package.md). The focused suite is:

```bash
python -m unittest tests.test_model_package tests.test_model_export tests.test_checkpoint_training tests.test_config tests.test_runtime tests.test_nats_requests tests.test_rdkit_api tests.test_training_pipeline
```

## Containers and planned runtime boundary

[Dockerfile.test](Dockerfile.test) and [Dockerfile.staging](Dockerfile.staging) both start
the service through module `main`. The test Dockerfile currently starts the application;
its name does not mean it runs the test suite. The existing
[Compose](docker_mt21/docker-compose.yml) supplies `PY_ENV=test` but not a complete startup
configuration. Container consolidation and validation from a clean context remain pending.

The target application artifact is an inference-service image containing only runtime
code/dependencies and the compatible model package: weights, network definition,
fingerprint settings, labels and thresholds. Training code/dependencies, datasets,
notebooks and historical model copies stay out of the final staging/production image.
This boundary is a requirement; the current `COPY . .` Dockerfiles do not yet enforce it.

The coordinated roadmap and design inventory are maintained in MercurionWeb. The linked
baseline is immutable; later changes should be read on the corresponding Web work branch
until integration:

- [Container and delivery roadmap](https://github.com/giuliomarinelli/MercurionWeb/blob/fba1f1fdbbf56f4a64fec85a0b22c8172f3cf80f/docs/infrastructure/tox21-container-roadmap.md)
- [Design constraints and test inventory](https://github.com/giuliomarinelli/MercurionWeb/blob/fba1f1fdbbf56f4a64fec85a0b22c8172f3cf80f/docs/infrastructure/tox21-design-test-inventory.md)

## License

See [LICENSE.md](LICENSE.md).
