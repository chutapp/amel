#!/usr/bin/env bash
#
# Strip "Co-Authored-By: Claude ..." trailer from all commits via filter-branch,
# then delete and recreate chutapp/amel so the pre-strip SHAs are unreachable
# by URL. Same pattern as 2026-05-30-recreate-repo-pii-scrub.sh.
#
# Run from repo root, AFTER:
#   - filter-branch has already been run (local main carries the rewritten history).
#   - Local safety tag `pre-trailer-strip-backup` points at the pre-strip tip.
#   - `gh auth status` confirms the active account is `chutapp` (delete_repo scope).
#
set -euo pipefail

REPO="chutapp/amel"
DESCRIPTION="AMEL: Accumulated Message Effects on LLM Judgments — code, data, and analysis"
HOMEPAGE="https://arxiv.org/abs/XXXX.XXXXX"   # update once arXiv ID is live

echo "==> Sanity checks"
git rev-parse --abbrev-ref HEAD | grep -qx main || { echo "Not on main"; exit 1; }
git diff --quiet || { echo "Working tree dirty"; exit 1; }
git diff --cached --quiet || { echo "Index dirty"; exit 1; }

# Confirm rewritten main no longer contains the trailer
if git log --format=%B main | grep -iE "^Co-Authored-By:.*Claude" >/dev/null; then
  echo "Local main still mentions Claude trailer — filter-branch did not run? abort"
  exit 1
fi
echo "    local main clean of Claude trailer"

git rev-parse pre-trailer-strip-backup >/dev/null 2>&1 \
  || { echo "Safety tag 'pre-trailer-strip-backup' missing — refusing"; exit 1; }
echo "    safety tag present at $(git rev-parse --short pre-trailer-strip-backup)"

active="$(gh auth status 2>&1 | awk '/Logged in to github.com account/{acct=$(NF-1)} /Active account: true/{print acct; exit}')"
echo "    active gh account: $active"
[[ "$active" == "chutapp" ]] || { echo "Active gh account is not chutapp — run: gh auth switch --user chutapp"; exit 1; }

echo
read -r -p "About to DELETE github.com/${REPO} and recreate it. Type the repo name to confirm: " confirm
[[ "$confirm" == "$REPO" ]] || { echo "Mismatch — abort"; exit 1; }

echo
echo "==> Deleting $REPO"
gh repo delete "$REPO" --yes

echo
echo "==> Recreating $REPO"
gh repo create "$REPO" \
  --public \
  --description "$DESCRIPTION" \
  --homepage "$HOMEPAGE" \
  --disable-wiki

echo
echo "==> Pushing rewritten history"
git push -u origin main
# v1.0 release tag — pushed explicitly (we do NOT push the safety tag)
git push origin v1.0

echo
echo "==> Verifying remote head"
gh api "repos/${REPO}/commits/main" --jq '.sha, .commit.message' | head -3

echo
echo "Done. Manual follow-ups:"
echo "  - Re-enable branch protection on 'main'."
echo "  - Re-add About blurb / topics / social preview image."
echo "  - Update homepage URL once arXiv ID is live."
echo "  - Drop the local backup tag once happy: git tag -d pre-trailer-strip-backup"
