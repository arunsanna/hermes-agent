# Hermes ACP delegation join-barrier + never-zombie dispatch — implementation brief

## Problem (verified live, session `bd359ae9`, batch `deleg_c3e02742`)

On the ACP surface, Hermes dispatched 3 reviewers via `delegate_task(background=True)`,
then emitted its FINAL answer immediately and ended the turn. The children's results
never re-entered; the dispatch card is frozen `in_progress` forever. Two user-visible
defects:

1. **Answered before subagents completed** — parent finalizes without consolidating.
2. **Zombie dispatch card** — one agent shows "running" permanently.

Root cause: `background=True` is fire-and-forget by contract; the tool even tells the
model "Do not wait or poll — just continue" (`tools/delegate_tool.py:2848-2859`). The
re-entry rail (`process_registry.completion_queue` → the ACP watcher) is unreliable and,
when it fires, opens a NEW frame instead of closing the original card. Upstream built
durable completion delivery but wired it ONLY to CLI/gateway/tui_gateway — never to
`acp_adapter/` — so this is our surface to fix.

## Design: turn-end JOIN BARRIER at the ACP boundary (NOT in conversation_loop.py)

Principle: **A turn may not finalize while it still owes results to background children
it dispatched during that same turn.** When the model dispatches background children and
then tries to end the turn, the ACP layer JOINS them, folds their consolidated results
into the conversation, and RE-RUNS the agent once so it produces a single consolidated
answer. The dispatch card is closed. Genuine detach only survives if the model keeps
working (turn does not end) or the join times out (late results still arrive via the
existing watcher rail).

Why the ACP boundary: `agent.run_conversation` runs the whole turn in an executor and
`state.is_running` stays True across it — the watcher only drains a session's completions
when `is_running == False` (`acp_adapter/server.py:620-623`), so the barrier owns its
session's completions with NO race. `conversation_loop.py` is a 55-upstream-commit drift
hotspot and MUST NOT be touched.

## Verified code anchors (read these before editing)

- `acp_adapter/server.py`
  - `prompt()` at ~L1475. Turn runs via `agent.run_conversation(...)` inside
    `_run_agent()` (L1648-1730), executed at `result = await loop.run_in_executor(_executor, ctx.run, _run_agent)` (**L1742**).
  - Session key binding: `set_session_vars(session_key=session_id)` (**L1661**) — so
    async delegations dispatched this turn carry `session_key == session_id`. Key the
    barrier off `session_id` + a captured `turn_start_ts` (`time.time()` before the run).
    Do NOT thread a turn_id into delegate_tool.py.
  - Barrier goes AFTER L1742 returns and BEFORE `flush_open_tool_calls(...)` (**L1756**).
  - `is_running` set False at L1745-1746 (error path) and after (success path); keep it
    True during the barrier.
  - Existing watcher: `_async_delegation_watcher` / `_drain_completion_queue_once`
    (L562-634), `_notify_background_completion` (L636-678) — REUSE `format_process_notification`
    and the queue; do not duplicate the drain logic's routing.
- `tools/async_delegation.py`
  - `dispatch_async_delegation` (L168), `dispatch_async_delegation_batch` (L355);
    records in `_records` under `_records_lock` carry `session_key`, `status`, `is_batch`,
    `results`, `dispatched_at`. `_finalize`/`_finalize_batch` (L284/L470) push the
    completion event onto `process_registry.completion_queue`. `active_count()` (L140),
    `list_async_delegations()` (L526, strips `interrupt_fn`), `_reset_for_tests()` (L566).
- `tools/delegate_tool.py`
  - background branch L2761-2878; `note` L2848-2859; `background` schema desc ~L3417.
- `acp_adapter/tools.py`
  - `_is_async_background_dispatch` (L243), used in the tool-complete path (L1339) to keep
    the dispatch card `in_progress`; `make_step_cb`/`build_tool_complete`/`flush_open_tool_calls`.

## Deliverable contract (locked)

### Piece 0 — registry primitives (`tools/async_delegation.py`)
- Add a `threading.Event` `done_event` to each record; `_finalize` and `_finalize_batch`
  call `done_event.set()` AFTER pushing the completion event. Strip `done_event` from
  `list_async_delegations()` (like `interrupt_fn`).
- Optional `turn_id` param on both dispatch fns, stored on the record (nice-to-have; the
  barrier keys off session_key+since_ts, so turn_id is not required plumbing).
- `running_for_session(session_key: str, since_ts: float | None = None) -> list[dict]`:
  snapshot dicts (no non-serialisable fields) of records with `status=="running"`,
  matching `session_key`, and `dispatched_at >= since_ts` when given.
- `join(delegation_ids: list[str], timeout: float) -> dict` (e.g.
  `{"completed": [...], "pending": [...]}`): wait on each record's `done_event` sharing a
  single deadline; never hold `_records_lock` while waiting.
- Thread-safety preserved; all list/scan under `_records_lock`.

### Piece 1 — join barrier (`acp_adapter/server.py::prompt`)
- Capture `turn_start_ts = time.time()` just before L1742.
- After the run returns, before flush: `pending = running_for_session(session_id, turn_start_ts)`.
  Config gate `delegation.acp_join_same_turn` (default True). If disabled or `pending`
  empty → behave exactly as today.
- Else loop, max `delegation.acp_join_max_rounds` (default 3):
  1. `join([...ids], timeout=delegation.acp_join_timeout_seconds)` (default 180) — run the
     blocking join via `loop.run_in_executor` so the event loop is not blocked.
  2. Drain THIS session's `type=="async_delegation"` events from
     `process_registry.completion_queue` (requeue everything else, exactly like
     `_drain_completion_queue_once`), format each via `format_process_notification`, append
     to `state.history` as a `{"role":"user",...}` message.
  3. Re-run the agent to consolidate: reuse the `_run_agent` scaffolding (session vars,
     approval cb, callbacks) with a short internal continuation user message such as
     "Your background subagent(s) have completed; their results are above. Incorporate them
     and give your consolidated final answer." Set `result` to this run's result. Reuse the
     SAME `step_cb`/`message_cb`/etc. so the consolidated answer streams to the client.
  4. Recompute `pending` (a consolidation turn may spawn more); stop when empty or rounds
     exhausted.
- Timeout path: if `join` returns `pending`, do NOT hang — inject a brief
  "subagent(s) still running; results will arrive shortly" note, let the existing watcher
  deliver late completions as a continuation turn (current behavior), and mark cards per
  Piece 2. Bounded, never infinite.

### Piece 2 — never-zombie dispatch card (`acp_adapter/tools.py` + flush)
- Maintain a per-session map `delegation_id -> dispatch tool_call_id` (the dispatch tool
  result JSON carries `delegation_id`; record it where the dispatch card is created /
  `_is_async_background_dispatch` marks it in_progress). Thread the map into `prompt()`.
- When the barrier drains a completion, UPDATE the original dispatch card (by its
  tool_call_id) to `completed`/`failed` with the aggregated result — do not emit a new frame.
- `flush_open_tool_calls`: any dispatch card still `in_progress` after the barrier →
  terminal state. Joined+resolved → `completed`; genuine detach (join timed out or prior
  turn) → a distinct non-spinner terminal state; no result at all → `failed`
  ("subagent result not received"). NEVER leave `in_progress` past end_turn.

### Piece 3 — flag reframe (`tools/delegate_tool.py`, string-only)
- Rewrite the background dispatch `note` (L2848-2859) and the `background` schema
  description (~L3417): background = detached, results re-enter in a LATER turn; on the ACP
  surface, if you end your turn they will be awaited and consolidated. For work you must
  consolidate into THIS reply, prefer synchronous (default). Remove "Do not wait or poll."

## Scope boundaries (do not cross)
- Edit ONLY: `tools/async_delegation.py`, `acp_adapter/server.py`, `acp_adapter/tools.py`,
  `tools/delegate_tool.py` (strings only), plus their test files. New config keys read via
  the existing config loader used in delegate_tool.py (`_load_config`).
- DO NOT touch `agent/conversation_loop.py` (55-commit upstream drift hotspot).
- DO NOT modify the intent of the existing 14 stability patches (finalize-hook bound,
  codex watchdog fixes, py3.14 executor fix, the in_progress dispatch patch 3f4112524 —
  build ON it, the watcher 5797a340c — extend it).
- DO NOT run or restart any hermes-acp process; this is the isolated worktree.

## Quality gates (MANDATORY — TDD, run yourself, paste real output)
Test env (already built): from the worktree root, `.venv/bin/python -m pytest ... -q`.
- Strict TDD per piece: write the failing test FIRST, watch it fail, minimal code to pass.
- New tests required:
  - registry: `running_for_session` filtering (session/since), `join` completed vs timeout,
    `done_event` set after completion event enqueued.
  - barrier: same-turn running delegation → agent re-run happens and result consolidated;
    no pending → no re-run (unchanged path); join timeout → bounded, no hang, note injected.
  - never-zombie: dispatch card flipped to completed on drained completion; flush leaves NO
    in_progress card.
- Regression: these must stay green —
  `.venv/bin/python -m pytest tests/tools/test_async_delegation.py tests/acp_adapter/test_background_completion_watcher.py -q`
  (baseline = 25 passed). Also run any `tests/acp_adapter/` and `tests/tools/test_delegate*`.

## Report format (return this, be honest about gaps)
- file:line map of every change.
- new test names + PASS/FAIL with pasted `pytest -q` tail.
- regression suite result (pasted).
- exactly what was NOT exercised (e.g. real multi-round consolidation, real client
  streaming) and why.
- any scope boundary you had to approach and why.
