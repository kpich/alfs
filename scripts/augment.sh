#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Supported values: wikibooks, wikisource
SOURCE=wikibooks

make download SOURCE="$SOURCE"
make etl SOURCE="$SOURCE"
make seg
