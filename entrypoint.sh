#!/usr/bin/env bash
set -euo pipefail

mkdir -p instance

export FLASK_APP="wsgi:app"

python -m flask db upgrade

exec gunicorn wsgi:app