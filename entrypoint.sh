#!/usr/bin/env bash
set -e

mkdir -p instance

flask db upgrade

exec gunicorn wsgi:app