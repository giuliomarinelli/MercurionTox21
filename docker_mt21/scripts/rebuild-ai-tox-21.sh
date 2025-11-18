#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "🔄 Rebuilding mercurion_tox_21..."

docker compose stop mercurion_tox_21
docker compose build mercurion_tox_21 --no-cache
docker compose up -d mercurion_tox_21
