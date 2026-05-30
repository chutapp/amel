#!/usr/bin/env bash
#
# Nuclear PII scrub for chutapp/amel — delete and recreate the public repo
# so the pre-scrub commit messages are unreachable by any URL (force-push
# leaves them retrievable by SHA until GitHub GCs them).
#
# Run from the repo root, AFTER:
#   - The local main branch already contains the scrubbed history
#     (commit 6fbde2c "audit: pre-publication corrections" or later).
#   - You have a local safety tag (`pre-msg-rewrite-backup`) pointing at
#     the pre-scrub tip in case you need to recover anything.
#   - `gh auth status` shows you are logged in with delete_repo scope.
#
# Verify scope first:
#   gh auth refresh -h github.com -s delete_repo
#
set -euo pipefail

REPO="chutapp/amel"
DESCRIPTION="AMEL: Accumulated Message Effects on LLM Judgments — code, data, and analysis"
HOMEPAGE="https://arxiv.org/abs/XXXX.XXXXX"   # update once arXiv ID is live

echo "==> Sanity checks"
git rev-parse --abbrev-ref HEAD | grep -qx main || { echo "Not on main"; exit 1; }
git diff --quiet || { echo "Working tree dirty — commit or stash first"; exit 1; }
git diff --cached --quiet || { echo "Index dirty — commit or reset first"; exit 1; }

# Confirm the scrubbed messages are actually in HEAD
if git log --format=%B main | grep -iE "upwork|pakistan|serbia" >/dev/null; then
  echo "Local main still mentions PII strings — abort"; exit 1
fi
echo "    local main is clean of PII strings"

# Confirm backup tag exists locally
git rev-parse pre-msg-rewrite-backup >/dev/null 2>&1 \
  || { echo "Safety tag 'pre-msg-rewrite-backup' missing — refusing"; exit 1; }
echo "    safety tag present at $(git rev-parse --short pre-msg-rewrite-backup)"

echo
read -r -p "About to DELETE github.com/${REPO} and recreate it. Type the repo name to confirm: " confirm
[[ "$confirm" == "$REPO" ]] || { echo "Mismatch — abort"; exit 1; }

echo
echo "==> Deleting $REPO"
gh repo delete "$REPO" --yes

echo
echo "==> Recreating $REPO (public, no init)"
gh repo create "$REPO" \
  --public \
  --description "$DESCRIPTION" \
  --homepage "$HOMEPAGE" \
  --disable-wiki

echo
echo "==> Pushing scrubbed history"
git push -u origin main
git push origin --tags 2>/dev/null || true   # push CITATION-style release tags if any

echo
echo "==> Verifying remote head"
gh api "repos/${REPO}/commits/main" --jq '.sha, .commit.message' | head -3

echo
echo "Done. Old commit SHAs are now unreachable on github.com (subject to "
echo "GitHub's standard 90-day unreferenced-object GC for the deleted repo, "
echo "which here is irrelevant since the repo itself was deleted)."
echo
echo "Manual follow-ups:"
echo "  - Update arXiv homepage URL in this script + repo settings."
echo "  - Re-enable branch protection on 'main' (Settings -> Branches)."
echo "  - Re-add any topics / social preview image / About blurb."
echo "  - Drop the local safety tag once verified: git tag -d pre-msg-rewrite-backup"
