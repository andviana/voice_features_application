#!/usr/bin/env bash
set -euo pipefail

mkdir -p instance

export FLASK_APP="wsgi:app"

python -m flask db upgrade

exec gunicorn wsgi:app --bind 0.0.0.0:${PORT:-10000} --workers "${WEB_CONCURRENCY:-1}" --threads 4 --timeout 120