#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Supported values: wikibooks, wikisource, wikipedia, gutenberg, cc_news
#SOURCE=wikibooks
#SOURCE=wikisource
#SOURCE=wikipedia
#SOURCE=gutenberg
#SOURCE=cc_news
SOURCE=wikisource

make download SOURCE="$SOURCE"
make etl SOURCE="$SOURCE"
make seg
