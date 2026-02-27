#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --project . python -m alfs.viewer.app
