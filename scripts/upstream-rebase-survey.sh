#!/usr/bin/env bash
# Safe, read-only survey for the weekly upstream rebase.
# Shows: latest upstream tag, commits to absorb, exact conflicting files, and a
# per-patch "upstream touched these files N times" obsolescence heuristic.
# Makes NO changes. See docs/UPSTREAM-REBASE-PLAYBOOK.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BRANCH="arun/switchboard-hermes-runtime"

echo "== fetching upstream (origin) + fork (arunsanna) tags =="
git fetch origin --tags --prune >/dev/null 2>&1 || true
git fetch arunsanna --prune       >/dev/null 2>&1 || true

NEWTAG="$(git tag --list --sort=-creatordate 'v2026*' | head -1)"
BASE="$(git merge-base "$BRANCH" "$NEWTAG")"
CURTAG="$(git describe --tags --abbrev=0 "$BRANCH" 2>/dev/null || echo '<none>')"

echo
echo "== targets =="
echo "  our branch base : $CURTAG ($(git rev-parse --short "$BASE"))"
echo "  latest tag      : $NEWTAG ($(git log -1 --format='%ci' "$NEWTAG" | cut -d' ' -f1))"
echo "  upstream commits to absorb (base..$NEWTAG): $(git rev-list --count "$BASE".."$NEWTAG")"
echo "  our patches replayed on top: $(git rev-list --count "$BASE".."$BRANCH")"

echo
echo "== conflict surface (merge-tree preview of $BRANCH onto $NEWTAG) =="
if git merge-tree --write-tree --name-only "$NEWTAG" "$BRANCH" >/tmp/_mt 2>&1; then
  echo "  CLEAN — no conflicts. Rebase should be trivial."
else
  grep -E '^CONFLICT|^[a-z].*\.py$' /tmp/_mt | grep -viE '^auto-merging' | sed 's/^/  /' || true
fi

echo
echo "== per-patch upstream-touch heuristic (high count = check for obsolescence) =="
for sha in $(git log --format=%H "$BASE".."$BRANCH"); do
  subj="$(git log -1 --format='%h %s' "$sha")"
  files="$(git show --stat --format='' "$sha" | awk -F'|' 'NF>1{print $1}' | grep -vE '^\s*(tests/|.*test_)' | tr -d ' ' | head -4)"
  n=0
  for f in $files; do
    [ -n "$f" ] || continue
    c="$(git rev-list --count "$BASE".."$NEWTAG" -- "$f" 2>/dev/null || echo 0)"
    n=$((n + c))
  done
  printf '  [%3d] %s\n' "$n" "$subj"
done

echo
echo "Next: follow docs/UPSTREAM-REBASE-PLAYBOOK.md Part A (rebase in a throwaway worktree)."
