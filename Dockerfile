# syntax=docker/dockerfile:1
# Linux amd64 CPU. Supply --build-context model_package=<exported package directory>.
FROM python:3.10.11-slim@sha256:fd86924ba14682eb11a3c244f60a35b5dfe3267cbf26d883fb5c14813ce926f1 AS runtime-dependencies
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements/build-linux-cpu.lock requirements/runtime-linux-cpu.lock ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --require-hashes -r build-linux-cpu.lock && \
    /opt/venv/bin/python -m pip install --require-hashes --no-build-isolation -r runtime-linux-cpu.lock && \
    /opt/venv/bin/python -m pip check

FROM runtime-dependencies AS runtime-source
WORKDIR /app
COPY main.py config.py runtime.py ./
COPY api/__init__.py api/inference.py api/rdkit.py api/pcp.py ./api/
COPY schemas/__init__.py schemas/schemas.py ./schemas/
COPY mercurion/__init__.py mercurion/model.py mercurion/model_spec.py mercurion/labels.py \
     mercurion/fingerprints.py mercurion/model_package.py ./mercurion/

FROM runtime-source AS validation
COPY requirements/training-linux-cpu.lock /build/
RUN /opt/venv/bin/python -m pip install --require-hashes --no-build-isolation -r /build/training-linux-cpu.lock && \
    /opt/venv/bin/python -m pip check
COPY mercurion/calibration.py mercurion/export_model.py mercurion/train.py \
     mercurion/preprocessing.py mercurion/early_stopping.py mercurion/focal_loss.py ./mercurion/
COPY tests ./tests
COPY scripts/run_tests.py ./scripts/run_tests.py
RUN /opt/venv/bin/python -B scripts/run_tests.py all && \
    /opt/venv/bin/python -c "from pathlib import Path; Path('/build/tests-passed').write_text('Self-contained unittest groups passed\n')"

FROM python:3.10.11-slim@sha256:fd86924ba14682eb11a3c244f60a35b5dfe3267cbf26d883fb5c14813ce926f1 AS runtime
ARG BUILD_VERSION
ARG BUILD_REVISION
LABEL org.opencontainers.image.source="https://github.com/giuliomarinelli/MercurionTox21" \
      org.opencontainers.image.version=$BUILD_VERSION \
      org.opencontainers.image.revision=$BUILD_REVISION
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TMPDIR=/tmp MODEL_PACKAGE_DIR=/app/model VERSION=$BUILD_VERSION
RUN python -c "import os,re; assert os.environ.get('BUILD_VERSION','').strip(), 'BUILD_VERSION required'; assert re.fullmatch('[0-9a-fA-F]{40}', os.environ.get('BUILD_REVISION','')), 'Full BUILD_REVISION required'" && \
    groupadd --gid 10021 mercurion && \
    useradd --uid 10021 --gid 10021 --no-create-home --shell /usr/sbin/nologin mercurion
WORKDIR /app
COPY --from=runtime-dependencies /opt/venv /opt/venv
COPY --from=runtime-source /app /app
COPY --from=model_package /weights.pt /thresholds.json /manifest.json ./model/
RUN rm -rf /opt/venv/lib/python3.10/site-packages/pip /opt/venv/lib/python3.10/site-packages/pip-*.dist-info /opt/venv/bin/pip* && \
    chmod 0555 /app/model && chmod 0444 /app/model/weights.pt /app/model/thresholds.json /app/model/manifest.json
USER 10021:10021
RUN python -c "from runtime import load_inference_runtime; m,d,p=load_inference_runtime(package_dir='/app/model'); r=p('CCO',m,d); assert len(r)==4 and all(0<=v['probability']<=1 for v in r.values())"
CMD ["python", "-m", "main"]

FROM runtime AS staging
ENV PY_ENV=staging
LABEL org.mercurion.environment=staging

FROM runtime AS test
ENV PY_ENV=test
LABEL org.mercurion.environment=test
COPY --from=validation /build/tests-passed /usr/share/mercurion/tests-passed

FROM runtime AS production
ENV PY_ENV=production
LABEL org.mercurion.environment=production
