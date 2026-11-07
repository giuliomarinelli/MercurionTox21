# Dependency profiles — T21-03

The supported clean-install target verified here is **Linux x86_64, CPython 3.10.11,
CPU**. The existing Docker base is pinned for verification to
`python:3.10.11-slim@sha256:fd86924ba14682eb11a3c244f60a35b5dfe3267cbf26d883fb5c14813ce926f1`.
This is a tested baseline, not a claim about all Python/platform combinations or GPU.

| File | Purpose |
| --- | --- |
| `runtime.in` | Direct service/inference dependencies at their existing versions |
| `training.in` | Runtime plus local training, calibration/export and evaluation |
| `runtime-linux-cpu.lock` | Full 23-distribution runtime closure with hashes |
| `training-linux-cpu.lock` | Full 39-distribution training/export closure with hashes |
| `build-linux-cpu.lock` | Pinned setuptools backend required by the NATS source distribution |

The original root `requirements.txt` is retained as a legacy development/research
umbrella and a constraint source for generating these locks. It is not the runtime
profile. In particular, the optional DeepChem dataset-download script is not part of
the tested runtime or core training/export profiles; existing research dependencies
are not silently removed or declared covered by these checks.

## Install in a new matching environment

Use a fresh container/virtualenv for this target; do not replace a working local environment
to test it. From the repository root, with the chosen environment's Python:

```bash
python -m pip install --require-hashes -r requirements/build-linux-cpu.lock
python -m pip install --require-hashes --no-build-isolation -r requirements/runtime-linux-cpu.lock
python -m pip check
```

For local model development/export, use `training-linux-cpu.lock` in the second command
instead. This lock includes the runtime closure too. NATS 2.9.0 is distributed as source;
the source hash and its build backend are pinned. Do not drop `--no-build-isolation`:
otherwise pip may fetch an unpinned isolated build backend. CPU Torch comes from the
official PyTorch CPU index; other packages use PyPI. All selected distributions must
match the hashes in the lock.

The verification installed from the downloaded distributions **offline** using
`--no-index --find-links <download-directory>` in both commands. No compiler, extra
unlocked build dependency or training library was needed in the runtime environment.
Native Torch/RDKit imports and a prediction with the real package were verified there.
The application Dockerfiles still use the legacy requirements until T21-04 adopts
these profiles; the validation images are not a completed application image release.

## Regenerate deliberately on the supported target

Use separate empty download directories for runtime and training. Preserve the old pins
as constraints; an intentional version change requires its own review and verification:

```bash
python -m pip download --prefer-binary --dest /tmp/t21-runtime-downloads \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements/runtime.in -c requirements.txt 'torch==2.13.0+cpu'
python scripts/lock_requirements.py /tmp/t21-runtime-downloads \
  requirements/runtime-linux-cpu.lock --bootstrap-output requirements/build-linux-cpu.lock
```

Repeat with `training.in`, another empty directory and output `training-linux-cpu.lock`.
The script hashes downloaded wheels/source archives without extracting or importing them.
Hash generation records integrity of the selected inputs; the clean install, `pip check`
and native tests are still required. This does not promise byte-identical image layers or
NATS wheel timestamps across builds.

The original Windows virtualenv remains unchanged. Its broader scientific import checks
stalled inside native pandas/scikit-learn modules; the clean Linux checks passed instead.
Windows dependency remediation is not claimed complete by these platform-specific locks.
