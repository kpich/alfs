#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure ollama is running
if ! curl -sf http://localhost:11434/ > /dev/null 2>&1; then
    echo "Starting ollama..."
    ollama serve &>/dev/null &
    # Wait up to 10s for it to become ready
    for i in $(seq 1 10); do
        sleep 1
        curl -sf http://localhost:11434/ > /dev/null 2>&1 && break
        [ "$i" -eq 10 ] && { echo "ERROR: ollama did not start"; exit 1; }
    done
fi

nextflow run nextflow/relabel.nf "$@"
