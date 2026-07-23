# Phase 0 — Hermes STOP correctness (hermes-acp fork)

Goal: a turn that COOPERATIVELY returns must ALWAYS free the session (`state.is_running=False`)
and drain the queue — on normal, cancelled, or errored exit — and the join-barrier must not run
on a cancelled turn. Fixes the stuck-queue-after-STOP for the common case + the barrier regression.

## Verified anchors — `acp_adapter/server.py` (read them)
- `prompt()` runs the turn via `_run_agent` in an executor; executor result at **~L1806** (`result = await loop.run_in_executor(...)`).
- **My join-barrier: L1825-1940** (`running_for_session`, `join`, `_run_agent(continuation)` re-run) — has NO cancel guard.
- `cancelled = bool(state.cancel_event and state.cancel_event.is_set())` at **L1990** (currently AFTER the barrier).
- `state.is_running = False` at **L2053** — NAKED in the success path, NOT in a finally → leaks True if anything between L1806 and L2053 hangs/throws.
- Queue drain at **L2056-2069** (`while ... state.queued_prompts.pop(0) ... await self.prompt(...)`).
- `cancel_event.set()` at **L1435** (the STOP path); `state.queued_prompts` appended at **L1590**.

## Changes (TDD — failing test first for each)

### P0.1 — Guarantee is_running reset + queue drain in a `finally`
Restructure the post-executor body (from just after L1806 through the drain) so that
`state.is_running = False`, `state.current_prompt_text = ""`, AND the queue-drain loop run in a
`finally` (or an equally guaranteed path) — reached no matter what the barrier/response code does
(exception, early logic). No path between the executor return and that finally may leave
`is_running` True. Keep the existing drain semantics (pop queued prompts, recursively `self.prompt`).
- Test: monkeypatch the barrier region to raise → assert `state.is_running is False` afterward AND queued prompts drained.

### P0.2 — Barrier cancel guard
Immediately before the barrier work (L1825), compute `cancelled = bool(state.cancel_event and state.cancel_event.is_set())`.
If cancelled: SKIP the entire barrier (no `join`, no `_run_agent` re-run) and call
`tools.async_delegation.interrupt_all("user cancelled")` (or per-session interrupt) so background
subagents from the cancelled turn stop. Also re-check cancel between join rounds and abort the loop.
- Test: with `cancel_event` set and a pending delegation, assert the barrier does NOT re-run the agent and DOES interrupt.

### P0.3 — Cancel still drains the follow-ups
On the cancelled path, the interrupted turn is dropped but the queued prompts the user typed after
STOP must still drain (the P0.1 finally-drain handles this — just ensure the cancelled branch reaches it).
- Test: cancel a turn that has 2 queued prompts → both drain (run) afterward.

## Gate
`.venv/bin/python -m pytest tests/acp_adapter/ tests/tools/test_async_delegation.py -q` (baseline 45+20 green).
Strict TDD: RED first (paste the failing tail), minimal GREEN, no regressions.

## Scope
ONLY `acp_adapter/server.py` + its tests. Do NOT change the barrier's core join/consolidate logic
beyond the cancel guard. Do NOT touch `agent/conversation_loop.py`. Do NOT run/restart any live process.

## Report
file:line map, new test names + pasted pytest tails, regression result, honest list of un-exercised paths.
