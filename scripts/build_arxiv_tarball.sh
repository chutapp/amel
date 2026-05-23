#!/usr/bin/env bash
# Build the arXiv submission tarball.
# Run from the repository root: bash scripts/build_arxiv_tarball.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# Strip macOS resource forks so arXiv does not have to clean up `._*` sidecars.
export COPYFILE_DISABLE=1
tar --no-xattrs -czf amel-arxiv.tar.gz \
    -C paper \
    main.tex \
    main.bbl \
    references.bib \
    qualitative_examples.tex \
    figures
echo "Built amel-arxiv.tar.gz ($(du -h amel-arxiv.tar.gz | cut -f1))"
echo "Contents:"
tar -tzf amel-arxiv.tar.gz
