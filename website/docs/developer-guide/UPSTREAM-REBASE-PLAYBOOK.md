# Upstream Rebase Playbook (Hermes fork)

How to keep this fork current with NousResearch upstream **without losing our patches**.
Model: our deployment branch = **latest upstream release tag + our patches rebased on top**.
Run this weekly (or as often as you want). The first catch-up (v0.18.0 → v0.19.0) is done;
each subsequent rebase is small because `rerere` replays prior conflict resolutions.

---

## Constants (this machine)

| Thing | Value |
|---|---|
| Fork repo (live source) | `/Users/jarvis_arunlab/code/research-lab/.hermes/acp-adapter` |
| Deployment branch (LIVE) | `arun/switchboard-hermes-runtime` |
| Remotes | `origin` = NousResearch upstream · `arunsanna` = our GitHub fork |
| Current base tag | `v2026.7.20` (v0.19.0) — bump this each rebase |
| Sanctioned Python | **3.12** (pyproject caps `<3.14`; do NOT use 3.14) |
| Env build | `uv sync --extra acp --extra dev` |
| Live spawn path | `~/.local/bin/hermes-acp` → `.venv/bin/hermes-acp` (editable install → this source tree) |
| Test suite | `.venv/bin/python -m pytest tests/tools/test_async_delegation.py tests/acp_adapter/ tests/tools/test_delegate.py -q` |
| `rerere` | enabled — records/auto-replays conflict resolutions |

**Golden rules**
- Rebase onto a **release TAG**, never onto `origin/main` HEAD (main moves mid-flight).
- Do the rebase in a **throwaway worktree**. Never rebase the live branch in place.
- **Drop a patch only** if upstream fixed the same issue — and record the superseding commit.
- Upstream historically ignores `acp_adapter/` — never drop ACP-surface patches as "upstream fixed it".
- Promote to the live branch **only after** tests + a spawn smoke + your live backend check pass.

---

## 0. One command to survey (safe, read-only)

```bash
cd /Users/jarvis_arunlab/code/research-lab/.hermes/acp-adapter
./scripts/upstream-rebase-survey.sh
```

Prints: latest upstream tag, how many commits you'd absorb, the **exact conflicting files**
(via `git merge-tree`), and a per-patch "upstream touched these files N times" heuristic
(high N = check for obsolescence). No changes made.

---

## A. Rebase (in a throwaway worktree)

```bash
cd /Users/jarvis_arunlab/code/research-lab/.hermes/acp-adapter
git fetch origin --tags --prune && git fetch arunsanna --prune

# 1. Pick the newest stable tag
NEWTAG=$(git tag --list --sort=-creatordate 'v2026*' | head -1); echo "target: $NEWTAG"
BASE=$(git merge-base arun/switchboard-hermes-runtime "$NEWTAG")   # our current base

# 2. Throwaway worktree + branch off the live branch
WT=../acp-adapter-rebase-$NEWTAG
git worktree add -b rebase/$NEWTAG "$WT" arun/switchboard-hermes-runtime
git -C "$WT" config rerere.enabled true
git -C "$WT" config rerere.autoupdate true

# 3. Replay our patches onto the new tag
cd "$WT"
git rebase --onto "$NEWTAG" "$BASE" HEAD
```

**When it stops on a conflict:**
- `rerere` auto-applies any resolution you've done before → just `git add -A && git rebase --continue`.
- New conflict → resolve preserving OUR intent. If a file was rewritten upstream, **read the
  upstream version first** (`git show $NEWTAG:path/to/file.py`).
- Patch is now redundant (upstream fixed the same issue): `git rebase --skip`, then note the
  superseding commit in the commit log / this file's changelog.
- Stuck: `git rebase --abort` and survey again.

---

## B. Keep / Drop / Rework decision

| Situation | Action |
|---|---|
| Upstream shipped a fix for the SAME issue (cite the commit) | **DROP** (`git rebase --skip`) |
| Upstream moved/rewrote the file but did NOT fix our issue | **REWORK** onto their new structure |
| Upstream never touched the area (esp. `acp_adapter/`) | **KEEP** as-is |

Find whether upstream addressed something:
```bash
git log "$BASE".."$NEWTAG" --oneline --grep -Ei 'KEYWORD' -- path/to/file.py
```

---

## C. Validate the rebased worktree

```bash
# still in the worktree
uv venv --python 3.12 && uv sync --extra acp --extra dev
.venv/bin/python -m pytest tests/tools/test_async_delegation.py tests/acp_adapter/ tests/tools/test_delegate.py -q
.venv/bin/python -c "from acp_adapter.server import HermesACPAgent; print('import ok')"
# spawn smoke (waits ~10s for MCP servers to load — that's normal):
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}' \
  | HERMES_HOME=/Users/jarvis_arunlab/code/research-lab/.hermes/home timeout 20 .venv/bin/hermes-acp | head -1
```
All green? Continue. Otherwise fix in the worktree until green.

---

## D. Promote to LIVE

```bash
cd /Users/jarvis_arunlab/code/research-lab/.hermes/acp-adapter
STAMP=$(date +%Y%m%d)

# 1. Backup for instant rollback
git branch backup/switchboard-hermes-runtime-pre-$STAMP arun/switchboard-hermes-runtime

# 2. Point the live branch at the rebased content
git checkout arun/switchboard-hermes-runtime
git reset --hard rebase/$NEWTAG

# 3. Rebuild the LIVE env on the new version
uv sync --extra acp --extra dev

# 4. Spawn smoke on the live binary (see Part C smoke command)

# 5. Clean up the worktree
git worktree remove "$WT" --force
```

**Then:** restart your running Hermes sessions (they hold old code in memory) and start a
**new** session to live-validate the backends you care about (Codex/Terra/Z.AI/Claude/etc.).

---

## E. Rollback (if the promotion misbehaves)

```bash
cd /Users/jarvis_arunlab/code/research-lab/.hermes/acp-adapter
git reset --hard backup/switchboard-hermes-runtime-pre-<STAMP>
uv sync --extra acp --extra dev      # rebuild env back to the old version
# restart Hermes sessions
```

---

## F. Shrink the patch set (makes every future rebase cheaper)

The fewer patches we carry, the less there is to replay. After each rebase:
- **Upstream our fixes** — open PRs to NousResearch for patches that aren't Switchboard-specific
  (we already have PR #68950 for the async dispatch fix). Every accepted PR = one fewer patch.
- Re-check KEEP patches for obsolescence each cycle; drop as upstream catches up.

---

## G. Optional: delegate the grind to a workhorse

For a big catch-up, hand Parts A–C to Codex in the worktree (it resolves conflicts + tests):
```bash
codex exec -C "$WT" -m gpt-5.6-sol -c model_reasoning_effort=high \
  -c model_auto_compact_token_limit=900000 -s danger-full-access < /dev/null \
  "Rebase our patches onto $NEWTAG per docs/UPSTREAM-REBASE-PLAYBOOK.md Parts A-C. Drop only
   patches upstream superseded (cite the commit); rework rewritten files; keep acp_adapter work.
   Rebuild env (python 3.12, uv sync --extra acp --extra dev), run the test suite + spawn smoke.
   Only this worktree. Report a KEPT/DROPPED/REWORKED table + test tails."
```
You still gate + promote (Parts D–E) yourself.

---

## Gotchas we hit (don't relearn them)

- **`.venv` can silently rebuild to 3.12 WITHOUT the `acp` extra** → every new session crashes on
  `import acp`. Fix: `uv sync --extra acp --extra dev` (or `uv pip install -e ".[acp]"`).
- **Spawn smoke looks like it hangs** — it's loading MCP servers (~7-10s). Wait 20s. A
  `robinhood-trading` 401 in stderr is a pre-existing MCP auth issue, unrelated.
- **After a rebase the fork branch has rewritten history** → pushing needs `git push --force-with-lease arunsanna arun/switchboard-hermes-runtime`.
- **Running Hermes procs keep old code in memory** — only NEW sessions pick up a promotion.
- **codex exec blocks on stdin** if launched without `< /dev/null`.
- **Never target Python 3.14** — pyproject excludes it (no cp314 wheels for Rust transitives).
