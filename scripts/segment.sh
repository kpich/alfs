#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nextflow run nextflow/segment.nf "$@"
