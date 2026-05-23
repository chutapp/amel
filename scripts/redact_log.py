"""One-shot redaction script for data/raw/experiment.log.

Replaces local home directories, Python install paths, and repository paths
with neutral placeholders so the log can be published without leaking
host-specific information. Applied once before the public release; kept in
the repository for transparency.

Run:  python scripts/redact_log.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("data/raw/experiment.log")


def main() -> None:
    text = SRC.read_text()
    text = re.sub(r"/Users/[^/]+/(?:[^\s\"']+/)*?(?=research/)", "<HOME>/", text)
    text = re.sub(r"/Users/[^/]+/code/[^/]+/research/[^/\s'\"]+", "<REPO>", text)
    text = re.sub(r"/Users/[^/]+/\.pyenv/versions/\d+\.\d+\.\d+", "<PYTHON>", text)
    text = re.sub(r"/Users/[^/\s\"']+", "<HOME>", text)
    SRC.write_text(text)
    remaining = len(re.findall(r"/Users/[a-z]+|MacBook-Pro", text))
    print(f"Redacted. Remaining sensitive patterns: {remaining}")


if __name__ == "__main__":
    main()
