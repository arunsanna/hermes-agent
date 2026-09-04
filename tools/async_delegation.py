#!/usr/bin/env python3
"""Async (background) delegation registry behind ``delegate_task(background=true)``.

The parent dispatches a subagent on a module-level daemon executor and returns a handle
immediately. On completion a ``type="async_delegation"`` event (self-contained task-source
block) is pushed onto the SHARED ``process_registry.completion_queue`` the CLI/gateway drain
while idle, so results surface as a NEW turn (never mid-turn) and inherit its de-dup and
crash-recovery wiring. Only the async lifecycle lives here; the child run is an injected ``runner``."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

# Backward-compatible import retained for ACP visibility callers/tests while
# the implementation lives in the shared daemon-pool module.
_DaemonThreadPoolExecutor = DaemonThreadPoolExecutor

# ── Module-level state ──────────────────────────────────────────────────────
# Persistent daemon executor (never a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat async); daemon workers can't hang a hard exit.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.Lock()
# delegation_id -> record dict; kept for the run plus a short completed tail.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# Completed records retained (in memory and in the ledger) for status queries.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# Cap retried deliveries so an unroutable row converges to terminal 'dropped'.
_MAX_DELIVERY_ATTEMPTS = 8
# Pending completions older than this are dropped on restart replay instead of
# re-run as a full-context turn; 48h keeps weekend results deliverable.
_MAX_COMPLETION_REPLAY_AGE_S = 48 * 3600.0
_DB_LOCK = threading.Lock()

# ── Stale-delegation detection (progress-based, on by default) ──────────────
# A runner wedged before returning never reaches its finalizer, so it would show
# "dispatched" forever. No wall-clock timeout (heavy work must never be killed for
# taking long): one monitor thread samples per-dispatch PROGRESS via an injected
# ``progress_fn``; a frozen child is interrupted, given a grace window to unwind via
# the normal finalize path, and only force-finalized (terminal ``stalled`` event) if
# it never returns. Thresholds mirror delegate_tool's sync heartbeat monitor.
_STALE_CHECK_INTERVAL = 30.0
_STALE_IDLE_SECONDS = 450.0
_STALE_IN_TOOL_SECONDS = 1200.0
_STALL_GRACE_SECONDS = 120.0

_monitor_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()

_LIVE_STATES = {"running", "stalling", "finalizing"}
_ACTIVE_STATES = ("running", "stalling")
# Required ACP batches have a controller-owned lifecycle that is intentionally
# separate from the legacy detached-completion rail.  Keep their live states
# visible to capacity/lineage checks while legacy listings filter them out.
# Terminal ``stalled`` is deliberately absent: once force-finalized it must
# release async capacity. Required supervision additionally contributes
# ``queued`` and ``slow`` before terminalization.
_LIVE_DELEGATION_STATES = frozenset({"queued", "running", "slow", "stalling", "finalizing"})
_REQUIRED_TERMINAL_STATES = frozenset({"completed", "failed", "error", "timeout", "cancelled", "interrupted"})
# Routing origin persisted at dispatch so a restart-recovered completion can
# reconstruct a full SessionSource (scope_id drives relay tenant egress).
_ROUTING_KEYS = ("scope_id", "user_id", "user_name")
# Structured stall metadata — additive, present only on stall finalizations.
_STALL_META_KEYS = ("stalled_after_quiet_seconds", "stall_threshold_seconds", "stall_phase", "stall_grace_seconds")
# Private stall bookkeeping on the record -> public field in list_async_delegations().
_STALL_FIELD_MAP = (("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                    ("_stall_threshold_seconds", "stall_threshold_seconds"), ("_stall_in_tool", "stall_in_tool"))


# ── Durable ledger (state.db / async_delegations) ───────────────────────────
def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()  # don't leak the connection on PRAGMA/DDL failure
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state_repair import apply_durability_barriers
    # Preserve the journal mode SessionDB configured on state.db: forcing WAL from
    # every short-lived connection collides with live transcript/FTS writers.
    apply_durability_barriers(conn)
    conn.execute("""CREATE TABLE IF NOT EXISTS async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            origin_session_id TEXT NOT NULL DEFAULT ''
        )""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    # origin_session_id: raw api_server session id of the ORIGINATING request
    # (wake target); without it restart-recovered completions are unroutable there.
    for name, sql_type in (("owner_pid", "INTEGER"), ("owner_started_at", "INTEGER"), ("task_json", "TEXT"),
                           ("delivery_claim", "TEXT"), ("delivery_claimed_at", "REAL"), ("origin_session_id", "TEXT")):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it (``with conn:``
    alone leaks the connection and WAL/SHM fds until GC).

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the transaction; they do not
    close the connection. Using ``with _connect()`` alone therefore leaks a connection — and its WAL/SHM
    file descriptors — on every durable dispatch, completion, and delivery-claim, deferring the close to the
    garbage collector. On a long-running gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of
    this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _capture_routing_origin() -> Dict[str, Any]:
    """Snapshot scope_id/user_id/user_name on the PARENT thread (the daemon worker
    has no contextvars) so a restart-replayed completion can rebuild a SessionSource.
    Best-effort: empty values are omitted."""
    try:
        from gateway.session_context import get_session_env
        return {k: v for k in _ROUTING_KEYS if (v := get_session_env(f"HERMES_SESSION_{k.upper()}", ""))}
    except Exception:  # noqa: BLE001 - routing origin is additive, never fatal
        return {}


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(os.getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch", *_ROUTING_KEYS)
        if key in record}
    with _DB_LOCK, _transaction() as conn:
        conn.execute("""INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""), record.get("origin_ui_session_id", ""),
             record.get("parent_session_id"), record["dispatched_at"], now, os.getpid(), owner_started_at,
             json.dumps(task_payload), record.get("origin_session_id", "")))
    _prune_durable_records()


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    cutoff = time.time() - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?", (cutoff,))
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')").fetchone()[0]
        if terminal_count > _MAX_RETAINED_COMPLETED:
            conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""", (terminal_count - _MAX_RETAINED_COMPLETED,))
        pending_count = conn.execute("""SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'""").fetchone()[0]
        if pending_count > _MAX_DURABLE_PENDING:
            conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT ?
                   )""", (pending_count - _MAX_DURABLE_PENDING,))


def _persist_completion(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute("""UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=?""",
            (event.get("status", "completed"), event.get("completed_at", now), now,
             json.dumps(event), json.dumps(result), event["delegation_id"]))


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
    now, recovered = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute("""SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id
               FROM async_delegations WHERE state IN ('running','finalizing')""").fetchall()
        for row in rows:
            delegation_id, session_key, origin_ui, parent_id, dispatched_at, pid, started, task_json, origin_sid = row
            if pid and _pid_exists(int(pid)) and (started is None or get_process_start_time(int(pid)) == int(started)):
                continue
            task = json.loads(task_json or "{}")
            event = {
                "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
                "origin_ui_session_id": origin_ui, "origin_session_id": origin_sid or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""), "goals": task.get("goals"),
                "context": task.get("context"), "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": "unknown", "summary": None,
                "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
                "dispatched_at": dispatched_at, "completed_at": now,
                **{k: task[k] for k in _ROUTING_KEYS if task.get(k)}}
            result = {"status": "unknown", "summary": None, "error": event["error"]}
            conn.execute("""UPDATE async_delegations SET state='unknown', completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""", (now, now, json.dumps(event), json.dumps(result), delegation_id))
            recovered += 1
    return recovered


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable pending completions as fresh turns after process start.
    Restored events are stamped ``restored=True`` in memory only: they came from a PREVIOUS
    process, so drains without an ownership filter must leave them for a consumer that can
    prove ownership. Rows older than ``_MAX_COMPLETION_REPLAY_AGE_S`` are terminally dropped
    instead of replaying a turn nobody is waiting on.

    Every restored event is stamped ``restored=True`` (in-memory only — the stamp is added after the durable
    payload is deserialized and is never persisted). Restored events originate from a *previous* process, so
    no consumer in THIS process implicitly owns them: drain paths that run without an ownership filter (the
    legacy single-session behavior) must leave them queued for a consumer that can positively prove
    ownership, otherwise a brand-new session adopts a dead session's delegation results seconds after boot
    (#64484).
    """
    recover_abandoned_delegations()
    now, restored = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id""").fetchall()
        for delegation_id, payload, completed_at, dispatched_at in rows:
            age_basis = completed_at or dispatched_at
            if age_basis and (now - age_basis) > _MAX_COMPLETION_REPLAY_AGE_S:
                conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL,
                              updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""", (now, delegation_id))
                logger.warning("Async delegation %s: pending completion is %.1fh old "
                               "(cap %.1fh); terminally dropping the replay (result remains queryable).",
                               delegation_id, (now - age_basis) / 3600.0, _MAX_COMPLETION_REPLAY_AGE_S / 3600.0)
                continue
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
            restored += 1
    return restored


def _update_delivery(sql: str, params: tuple) -> bool:
    """Run one UPDATE on the ledger; True iff exactly one row changed."""
    with _DB_LOCK, _transaction() as conn:
        return conn.execute(sql, params).rowcount == 1


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    return _update_delivery(
        """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
           WHERE delegation_id=? AND delivery_state!='delivered'""", (now, now, delegation_id))


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute("""UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, delegation_id, now - 300))
        return cur.rowcount == 1


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable delegation event; non-durable events need no token."""
    delegation_id = str(evt.get("delegation_id") or "") if evt.get("type") == "async_delegation" else ""
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    return claim_id if claim_completion_delivery(delegation_id, claim_id) else None


def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Release a failed delivery claim so another consumer may retry. Attempts are
    counted at claim time; once the budget is exhausted the row converges to
    terminal ``dropped`` (only pending rows replay on restart)."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS))
        if capped.rowcount == 1:
            logger.warning("Async delegation %s exhausted its %d delivery attempts; "
                           "marking terminally dropped (result remains queryable).",
                           delegation_id, _MAX_DELIVERY_ATTEMPTS)
            return True
        cur = conn.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""", (now, delegation_id, claim_id))
        return cur.rowcount == 1


def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion whose target is permanently gone (the
    spawning session ended at an explicit user boundary such as /new or reset).
    ``dropped`` — not ``delivered`` — keeps the ack honest; not ``pending`` keeps
    restart recovery from replaying it into a fail-closed drop forever."""
    return _update_delivery("""UPDATE async_delegations SET delivery_state='dropped',
                  updated_at=?, delivery_claim=NULL,
                  delivery_claimed_at=NULL
           WHERE delegation_id=? AND delivery_state='pending'
             AND delivery_claim=?""", (time.time(), delegation_id, claim_id))


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge acceptance for the consumer holding this claim."""
    now = time.time()
    return _update_delivery("""UPDATE async_delegations SET delivery_state='delivered',
                  delivered_at=?, updated_at=?, delivery_claim=NULL,
                  delivery_claimed_at=NULL
           WHERE delegation_id=? AND delivery_state='pending'
             AND delivery_claim=?""", (now, now, delegation_id, claim_id))


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    _event_delivery(complete_completion_delivery, evt, claim_id)


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    _event_delivery(release_completion_delivery, evt, claim_id)


def _event_delivery(fn, evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        fn(str(evt.get("delegation_id") or ""), claim_id)


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute("""SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      origin_session_id
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,)).fetchone()
    return None if row is None else {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1], "dispatched_at": row[2],
        "completed_at": row[3], "result": json.loads(row[4]) if row[4] else None, "delivery_state": row[5],
        "delivery_attempts": row[6], "origin_session_id": row[7] or ""}


# ── In-memory registry queries ──────────────────────────────────────────────
def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow, never shrink) the shared daemon executor; in-flight
    futures keep running on a replaced pool until it is collected."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            _executor = DaemonThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async-delegate")
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of live async delegation UNITS (a whole batch counts as ONE slot)."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") in _LIVE_DELEGATION_STATES)


def active_task_count() -> int:
    """Number of running child subagents (a batch of N contributes N; a batch with
    no goal list counts 1) — the truthful observability figure, unlike slots."""
    with _records_lock:
        return sum(
            len(r["goals"]) if r.get("is_batch") and isinstance(r.get("goals"), (list, tuple)) and r["goals"] else 1
            for r in _records.values() if r.get("status") in _LIVE_DELEGATION_STATES)


def _session_records(statuses, session_key: str, origin_ui_session_id: str, parent_session_id: str) -> list:
    """Records in ``statuses`` owned by a session: any non-empty selector claims the
    record — ``origin_ui_session_id`` (TUI tab), ``session_key`` (routing key at
    dispatch), or ``parent_session_id`` (spawner's durable id — the right one for
    gateway chats, whose session_key survives ``/new`` while the session id rotates)."""
    selectors = [(field, wanted) for field, wanted in (
        ("origin_ui_session_id", origin_ui_session_id), ("session_key", session_key),
        ("parent_session_id", parent_session_id)) if wanted]
    if not selectors:
        return []
    with _records_lock:
        return [r for r in _records.values() if r.get("status") in statuses
                and any(str(r.get(field) or "") == wanted for field, wanted in selectors)]


def has_live_for_session(session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "") -> bool:
    """Whether a session still owns any live (running/stalling/finalizing) delegation."""
    return bool(_session_records(_LIVE_STATES, session_key, origin_ui_session_id, parent_session_id))


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the cap. Caller holds ``_records_lock``."""
    terminal = _REQUIRED_TERMINAL_STATES | {"completed", "error", "stalled", "unknown"}
    completed = [
        (rid, r) for rid, r in _records.items()
        if r.get("status") in terminal and not (r.get("required") and r.get("consumed_at") is None)
    ]
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: max(0, len(completed) - _MAX_RETAINED_COMPLETED)]:
        _records.pop(rid, None)


def _current_origin_session_id() -> str:
    """Raw session id of the ORIGINATING api_server request, or ``""``. ``HERMES_SESSION_ID``
    is unsafe here: building the child agent calls ``set_current_session_id(child.session_id)``
    just before dispatch, so the wake would self-post into the subagent's own session. The
    request-scoped ``HERMES_SESSION_CHAT_ID`` (raw X-Hermes-Session-Id on api_server) survives
    child construction; on push platforms chat_id is a chat, not a session => ``""``."""
    try:
        from gateway.session_context import get_session_env
        is_api = get_session_env("HERMES_SESSION_PLATFORM", "") == "api_server"
        return (get_session_env("HERMES_SESSION_CHAT_ID", "") or "") if is_api else ""
    except Exception:
        return ""


# ── Dispatch ────────────────────────────────────────────────────────────────
def _single_crash(error: str, duration: float) -> Dict[str, Any]:
    return {"status": "error", "summary": None, "error": error, "api_calls": 0, "duration_seconds": duration}


def _batch_crash(error: str, duration: float) -> Dict[str, Any]:
    return {"results": [], "error": error, "total_duration_seconds": duration}


def _batch_status(combined: Dict[str, Any]) -> str:
    """Batch status: completed unless every child errored/was interrupted."""
    child_results = combined.get("results") or []
    ok = ("completed", "success")
    return "error" if child_results and all(r.get("status") not in ok for r in child_results) else "completed"


def _dispatch(
    *, delegation_id: str, goal: str, goals: Optional[List[str]], context: Optional[str],
    toolsets: Optional[List[str]], role: str, model: Optional[str], session_key: str,
    parent_session_id: Optional[str], runner: Callable[[], Dict[str, Any]], origin_ui_session_id: str,
    origin_session_id: str, interrupt_fn: Optional[Callable[[], None]], max_async_children: int,
    progress_fn: Optional[Callable[[], tuple]], capacity_error: str,
) -> Dict[str, Any]:
    """Shared dispatch core for single (``goals is None``) and batch units. Capacity check +
    record insert happen under ONE lock hold so concurrent dispatches can't both pass the check
    and exceed the cap. At capacity the dispatch is REJECTED (never queued) so a runaway model
    can't pile up unbounded background work."""
    is_batch = goals is not None
    label = " batch" if is_batch else ""
    classify = _batch_status if is_batch else (lambda r: r.get("status") or "completed")
    crash_result = _batch_crash if is_batch else _single_crash
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id, "goal": goal, **({"goals": list(goals)} if is_batch else {}),
        "context": context, "toolsets": list(toolsets) if toolsets else None, "role": role, "model": model,
        "session_key": session_key, "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id, "parent_session_id": parent_session_id,
        **_capture_routing_origin(),
        "status": "running", "dispatched_at": dispatched_at, "completed_at": None,
        "interrupt_fn": interrupt_fn, **({"is_batch": True} if is_batch else {}), "progress_fn": progress_fn,
        "done_event": threading.Event(),
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None, "_progress_ts": dispatched_at, "_interrupted_at": None}
    with _records_lock:
        running = sum(1 for r in _records.values() if r.get("status") in _LIVE_DELEGATION_STATES)
        if running >= max_async_children:
            return {"status": "rejected", "error": capacity_error}
        _records[delegation_id] = record
    _persist_dispatch(record)
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = classify(result)
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception(f"Async delegation{label} %s crashed", delegation_id)
            result = crash_result(f"{type(exc).__name__}: {exc}", round(time.time() - dispatched_at, 2))
        finally:
            _finalize(delegation_id, result, status)

    try:
        # Propagate the dispatching profile so the detached child resolves get_hermes_home() correctly.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        with _DB_LOCK, _transaction() as conn:
            conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))
        return {"status": "rejected", "error": f"Failed to schedule async delegation{label}: {exc}"}
    if progress_fn is not None:
        _ensure_stale_monitor()
    return {"status": "dispatched", "delegation_id": delegation_id}


def dispatch_async_delegation(
    *, goal: str, context: Optional[str], toolsets: Optional[List[str]], role: str, model: Optional[str],
    session_key: str, parent_session_id: Optional[str] = None, runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "", origin_session_id: str = "", interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN, progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.
    ``session_key``/``parent_session_id`` are captured on the parent thread (the worker carries
    no contextvars) and route the completion back to the spawning session.
    ``progress_fn() -> (token, in_tool)`` enables stale monitoring; omitted = unmonitored.
    Returns ``{"status": "dispatched", "delegation_id"}`` or ``{"status": "rejected", "error"}``."""
    delegation_id = _new_delegation_id()
    handle = _dispatch(
        delegation_id=delegation_id, goal=goal, goals=None, context=context,
        toolsets=toolsets, role=role, model=model, session_key=session_key,
        parent_session_id=parent_session_id, runner=runner,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn, max_async_children=max_async_children, progress_fn=progress_fn,
        capacity_error=(
            f"Async delegation capacity reached ({max_async_children} running). Wait for one to finish "
            "(its result will re-enter the chat), or run this task synchronously (background=false). "
            "Raise delegation.max_concurrent_children in config.yaml to allow more concurrent background subagents."))
    if handle["status"] == "dispatched":
        logger.info("Dispatched async delegation %s (session_key=%s): %s",
                    delegation_id, session_key or "<cli>", (goal or "")[:80])
    return handle


def dispatch_async_delegation_batch(
    *, goals: List[str], context: Optional[str], toolsets: Optional[List[str]], role: str, model: Optional[str],
    session_key: str, parent_session_id: Optional[str] = None, runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "", origin_session_id: str = "", interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN, delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
    parent_owner_token: str = "", required: bool = False, parent_turn_id: str = "",
    child_ids: Optional[List[str]] = None,
    child_interrupt_fn: Optional[Callable[[str], None]] = None,
    child_terminal_fn: Optional[Callable[[str, str, str], None]] = None,
    no_progress_timeout_seconds: float = 300.0,
    start_timeout_seconds: Optional[float] = None,
    in_flight_no_progress_timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit: ``runner`` runs the
    entire batch and returns the combined ``{"results": [...], "total_duration_seconds": N}``
    dict. The batch occupies ONE async slot (in-batch parallelism is bounded
    separately) and produces a SINGLE completion event carrying per-task ``results``."""
    if required:
        return _dispatch_required_batch(
            goals=goals, context=context, toolsets=toolsets, role=role, model=model,
            session_key=session_key, parent_session_id=parent_session_id, runner=runner,
            parent_owner_token=parent_owner_token, origin_ui_session_id=origin_ui_session_id,
            origin_session_id=origin_session_id, interrupt_fn=interrupt_fn,
            max_async_children=max_async_children, delegation_id=delegation_id,
            progress_fn=progress_fn, parent_turn_id=parent_turn_id, child_ids=child_ids,
            child_interrupt_fn=child_interrupt_fn, child_terminal_fn=child_terminal_fn,
            no_progress_timeout_seconds=no_progress_timeout_seconds,
            start_timeout_seconds=start_timeout_seconds,
            in_flight_no_progress_timeout_seconds=in_flight_no_progress_timeout_seconds,
        )
    delegation_id = delegation_id or _new_delegation_id()
    n = len(goals)
    combined_goal = goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    handle = _dispatch(
        delegation_id=delegation_id, goal=combined_goal, goals=goals, context=context,
        toolsets=toolsets, role=role, model=model, session_key=session_key,
        parent_session_id=parent_session_id, runner=runner,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn, max_async_children=max_async_children, progress_fn=progress_fn,
        capacity_error=(
            f"Async delegation capacity reached ({max_async_children} running). Wait for one to finish "
            "(its result will re-enter the chat), or raise delegation.max_concurrent_children in "
            "config.yaml to allow more concurrent background units."))
    if handle["status"] == "dispatched":
        logger.info("Dispatched async delegation batch %s (%d task(s), session_key=%s)",
                    delegation_id, n, session_key or "<cli>")
    return handle


# ── Finalization + completion events ────────────────────────────────────────
def _finalize(delegation_id: str, result: Any, status: str) -> None:
    """Atomically claim terminal delivery, push the completion event, then mark ``status``.
    ``result`` is a dict or a callable receiving the record snapshot (stall path). The record
    stays active ("finalizing") until durable persistence and queue publication finish; otherwise
    process shutdown can kill this daemon worker after status flips but before SQLite commits.
    A second call for the same id (late runner return after a forced stall) is a no-op."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in _ACTIVE_STATES:
            return
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["progress_fn"] = None  # stop stale-monitor sampling
        snapshot = dict(record)
    _push_completion_event(snapshot, result(snapshot) if callable(result) else result, status)
    with _records_lock:
        done_event = None
        if delegation_id in _records:
            _records[delegation_id]["status"] = status
            done_event = _records[delegation_id].get("done_event")
        _prune_completed_locked()
    if isinstance(done_event, threading.Event):
        done_event.set()


def _push_completion_event(record: Dict[str, Any], result: Dict[str, Any], status: str) -> None:
    """Push a type='async_delegation' event onto the shared completion queue. Batch records
    (``is_batch``) carry the per-task ``results`` list (plus live transcript paths, the
    full-fidelity record of each child's run) instead of a single summary. Best-effort: failure
    must not crash the worker, but it WOULD mean a silently-lost result, so we log loudly."""
    is_batch = bool(record.get("is_batch"))
    label = " batch" if is_batch else ""
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(f"Async delegation{label} %s finished but process_registry import failed; "
                     "result lost: %s", record.get("delegation_id"), exc)
        return
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()
    if is_batch:
        payload = {
            "is_batch": True, "results": result.get("results") or [],
            "live_transcripts": result.get("live_transcripts"), "error": result.get("error"),
            "total_duration_seconds": result.get("total_duration_seconds")}
    else:
        payload = {
            "summary": result.get("summary"), "error": result.get("error"), "api_calls": result.get("api_calls", 0),
            "duration_seconds": result.get("duration_seconds", round(completed_at - dispatched_at, 2))}
    evt = {
        "type": "async_delegation", "delegation_id": record.get("delegation_id"),
        # session_key routes back to the originating gateway session; "" => CLI.
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""), **({"goals": record.get("goals")} if is_batch else {}),
        "context": record.get("context"), "toolsets": record.get("toolsets"), "role": record.get("role"),
        "model": record.get("model") if is_batch else (result.get("model") or record.get("model")),
        "status": status, **payload, "dispatched_at": dispatched_at, "completed_at": completed_at,
        **({} if is_batch else {"exit_reason": result.get("exit_reason")}),
        **{k: record[k] for k in _ROUTING_KEYS if record.get(k)},
        **{k: result[k] for k in _STALL_META_KEYS if k in result}}
    _persist_completion(evt, result)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Async delegation{label} %s: failed to enqueue completion event; "
                     "result lost: %s", record.get("delegation_id"), exc)


# ── Stale monitor ───────────────────────────────────────────────────────────
def _ensure_stale_monitor() -> None:
    """Start (once) the stale-delegation monitor thread. One daemon thread serves
    every dispatch; it exits when no monitorable records remain and is restarted
    by the next dispatch with a ``progress_fn``."""
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_stale_monitor_loop, name="async-delegate-stale-monitor", daemon=True)
        _monitor_thread.start()


def _sweep_stale_locked(now: float):
    """One monitor pass over ``_records``; caller holds ``_records_lock``. Returns
    ``(stalled, expired, any_monitorable)``: newly-stalling ``(delegation_id, quiet_for, in_tool)``
    tuples, stalling ids past the grace window, and whether anything is left to monitor."""
    stalled, expired, any_monitorable = [], [], False  # (delegation_id, quiet_for, in_tool) / ids past grace
    for record in _records.values():
        status = record.get("status")
        if status == "stalling":
            any_monitorable = True
            if now - (record.get("_interrupted_at") or now) >= _STALL_GRACE_SECONDS:
                expired.append(record["delegation_id"])
            continue
        progress_fn = record.get("progress_fn")
        if status != "running" or progress_fn is None:
            continue
        any_monitorable = True
        try:
            token, in_tool = progress_fn()
        except Exception:
            # An unreadable child must not look permanently healthy —
            # keep the last timestamp running instead of refreshing it.
            token, in_tool = record.get("_progress_token"), False
        if token != record.get("_progress_token"):
            record.update(_progress_token=token, _progress_ts=now)
            continue
        quiet_for = now - (record.get("_progress_ts") or now)
        limit = _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
        if quiet_for >= limit:
            # Stall context feeds the terminal event and status listings.
            record.update(
                status="stalling", _interrupted_at=now, _stall_quiet_seconds=round(quiet_for, 2),
                _stall_threshold_seconds=limit, _stall_in_tool=bool(in_tool))
            stalled.append((record["delegation_id"], quiet_for, in_tool))
    return stalled, expired, any_monitorable


def _call_interrupt(fn, msg: str, *args) -> bool:
    """Invoke an ``interrupt_fn``; True on success, else debug-log ``msg`` (+ exc)."""
    if not callable(fn):
        return False
    try:
        fn()
        return True
    except Exception as exc:
        logger.debug(msg, *args, exc)
        return False


def _stale_monitor_loop() -> None:
    """Sweep running delegations for stalled progress. A changed progress token refreshes the
    record's timestamp; a frozen token past the idle/in-tool threshold marks the record
    ``stalling`` and calls ``interrupt_fn``; a ``stalling`` record still unreturned after the
    grace window is force-finalized with a terminal ``stalled`` event."""
    while not _monitor_stop.wait(_STALE_CHECK_INTERVAL):
        now = time.time()
        with _records_lock:
            stalled, expired, any_monitorable = _sweep_stale_locked(now)
        for delegation_id, quiet_for, in_tool in stalled:
            logger.warning("Async delegation %s made no progress for %.0fs "
                           "(in_tool=%s) — interrupting; grace window %.0fs",
                           delegation_id, quiet_for, in_tool, _STALL_GRACE_SECONDS)
            with _records_lock:
                fn = (_records.get(delegation_id) or {}).get("interrupt_fn")
            _call_interrupt(fn, "Async delegation %s stall interrupt failed: %s", delegation_id)
        for delegation_id in expired:
            _finalize(delegation_id, lambda rec, d=delegation_id: _stalled_result(d, rec), "stalled")
        if not any_monitorable:
            return


def _stalled_result(delegation_id: str, event_record: Dict[str, Any]) -> Dict[str, Any]:
    """Synthetic terminal result for a stalling delegation whose runner never returned."""
    completed_at = event_record.get("completed_at") or time.time()
    duration = round(completed_at - (event_record.get("dispatched_at") or completed_at), 2)
    error = (
        f"Async delegation {delegation_id} stalled: the detached subagent stopped making progress "
        "(no new API calls, tool activity, or streamed tokens), did not respond to interruption, and never "
        "produced a completion event. The worker may be wedged inside a model API call — this is a known "
        "failure mode of long-lived gateway processes (#60203). Re-dispatch the task if it is still needed.")
    logger.error("Async delegation %s force-finalized as stalled after %.0fs", delegation_id, duration)
    # Structured stall metadata lets parents/UIs distinguish a stall-monitor
    # kill from other failures without parsing the error string.
    stall_in_tool = event_record.get("_stall_in_tool")
    stall_meta = {
        "stalled_after_quiet_seconds": event_record.get("_stall_quiet_seconds"),
        "stall_threshold_seconds": event_record.get("_stall_threshold_seconds"),
        "stall_phase": "in_tool" if stall_in_tool else "idle" if stall_in_tool is not None else None,
        "stall_grace_seconds": _STALL_GRACE_SECONDS}
    if event_record.get("is_batch"):
        return {**_batch_crash(error, duration), **stall_meta}
    return {**_single_crash(error, duration), "status": "stalled", "exit_reason": "stalled", **stall_meta}


# ── Observability + control ─────────────────────────────────────────────────
def _children_activity_from_token(token: Any, now: float) -> Optional[List]:
    """Parse a progress token into per-child activity dicts (best-effort): delegate_tool
    emits one ``(api_call_count, current_tool, last_activity_ts)`` tuple per child;
    foreign token shapes degrade to ``None`` entries."""
    try:
        parts = list(token)
    except TypeError:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for part in parts:
        if not (isinstance(part, (list, tuple)) and len(part) >= 2):
            out.append(None)
            continue
        entry: Dict[str, Any] = {"api_calls": part[0], "current_tool": part[1]}
        if len(part) >= 3 and isinstance(part[2], (int, float)):
            entry["seconds_since_activity"] = round(max(0.0, now - float(part[2])), 1)
        out.append(entry)
    return out


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed) without callables or private
    monitor bookkeeping; adds computed live fields for UIs (``seconds_since_progress``,
    ``children_activity``/``in_tool`` sampled from ``progress_fn``) and stall context once tripped.

    Safe to call from any thread. See #51690.
    """
    now = time.time()
    samplers: Dict[str, Callable] = {}
    with _records_lock:
        items = []
        for r in _records.values():
            if r.get("required"):
                continue
            item = {
                k: v for k, v in r.items()
                if k not in {"interrupt_fn", "progress_fn", "done_event"} and not k.startswith("_")
            }
            status = r.get("status")
            if status in _ACTIVE_STATES:
                if r.get("_progress_ts"):
                    item["seconds_since_progress"] = round(now - r["_progress_ts"], 1)
                if callable(r.get("progress_fn")):
                    samplers[r["delegation_id"]] = r["progress_fn"]
            if status in ("stalling", "stalled"):
                for src, dst in _STALL_FIELD_MAP:
                    if r.get(src) is not None:
                        item[dst] = r.get(src)
            items.append(item)
    # Sample OUTSIDE the lock — progress_fn reads child-agent attributes and a
    # slow/broken sampler must not block every dispatch/finalize.
    for item in items:
        fn = samplers.get(item.get("delegation_id"))
        if fn is None:
            continue
        try:
            token, in_tool = fn()
        except Exception:
            continue
        activity = _children_activity_from_token(token, now)
        if activity is not None:
            item["children_activity"] = activity
        item["in_tool"] = bool(in_tool)
    return items


def _interrupt_records(targets: List[Dict[str, Any]], caller: str, reason: str, msg: str) -> int:
    """Call ``interrupt_fn`` on each record; log ``msg`` once; returns how many succeeded."""
    count = sum(
        _call_interrupt(r.get("interrupt_fn"), "%s: %s interrupt failed: %s", caller, r.get("delegation_id"))
        for r in targets)
    if count:
        logger.info(msg, count, reason)
    return count


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop (``/stop``, shutdown). Returns how
    many. The child still emits a completion event (status='interrupted') via the
    normal finalize path."""
    with _records_lock:
        targets = [r for r in _records.values() if r.get("status") in _ACTIVE_STATES]
    return _interrupt_records(targets, "interrupt_all", reason, "Interrupted %d async delegation(s) (%s)")


def interrupt_for_session(
    session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "", reason: str = "session_end",
) -> int:
    """Signal running async delegations owned by ONE ending session to stop (any
    selector matches, see ``_session_records``). Returns how many."""
    targets = _session_records(_ACTIVE_STATES, session_key, origin_ui_session_id, parent_session_id)
    return _interrupt_records(
        targets, "interrupt_for_session", reason, "Interrupted %d async delegation(s) for ending session (%s)")


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor + monitor."""
    global _executor, _executor_max_workers, _monitor_thread
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    _monitor_stop.set()
    with _monitor_lock:
        thread, _monitor_thread = _monitor_thread, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _records_lock:
        _records.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def active_for_session(origin_ui_session_id: str) -> int:
    """Number of live async delegations owned by one UI session."""
    if not origin_ui_session_id:
        return 0
    with _records_lock:
        return sum(
            1
            for r in _records.values()
            if r.get("status") in {"running", "stalling", "finalizing"}
            and str(r.get("origin_ui_session_id") or "")
            == origin_ui_session_id
        )
# ---- END PLUGIN-COMPAT ----
# Required ACP controller API ported from the pre-decomposition branch.  This
# remains in the lifecycle module (rather than a facade) because child agents
# and the ACP loop both call these functions directly.
def _clear_required_terminal_callbacks_locked(record: Dict[str, Any]) -> None:
    """Release child closures once controller terminalization wins."""
    record["interrupt_fn"] = None
    record["child_interrupt_fn"] = None
    record["child_terminal_fn"] = None


def _required_owner_matches(record: Dict[str, Any], parent_agent) -> bool:
    """Exact immutable-capability, same-turn ownership for required controls."""
    if not record.get("required") or parent_agent is None:
        return False
    record_token = str(record.get("parent_owner_token") or "")
    agent_token = str(
        getattr(parent_agent, "_required_delegation_owner_token", "") or ""
    )
    if record_token:
        return (
            bool(agent_token)
            and record_token == agent_token
            and str(record.get("parent_turn_id") or "")
            == str(getattr(parent_agent, "_current_turn_id", "") or "")
        )
    # Backward compatibility for in-memory records created by older embedders.
    return (
        str(record.get("parent_session_id") or "")
        == str(getattr(parent_agent, "session_id", "") or "")
        and str(record.get("parent_turn_id") or "")
        == str(getattr(parent_agent, "_current_turn_id", "") or "")
    )


def _required_record(parent_agent, delegation_id: str) -> Optional[Dict[str, Any]]:
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not _required_owner_matches(record, parent_agent):
            return None
        return record


def _required_public_snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    state = str(record.get("status") or "queued")
    children = record.get("child_supervision") or {}
    return {
        "delegation_id": record.get("delegation_id"),
        "required": True,
        "status": state,
        "terminal": state in _REQUIRED_TERMINAL_STATES,
        "consumed": record.get("consumed_at") is not None,
        "child_ids": list(record.get("child_ids") or []),
        "children": [
            _required_child_public_snapshot(child, now)
            for child_id in (record.get("child_ids") or [])
            for child in [children.get(str(child_id))]
            if isinstance(child, dict)
        ],
        "cancelled_child_ids": list(record.get("cancelled_child_ids") or []),
        "elapsed_seconds": round(
            max(0.0, now - float(record.get("dispatched_at") or now)), 2
        ),
        "last_liveness_at": record.get("last_liveness_at"),
        "last_meaningful_at": record.get("last_meaningful_at"),
        "last_activity": record.get("last_activity"),
        "current_tool": record.get("current_tool"),
        "progress_generation": int(record.get("progress_generation") or 0),
    }


def _required_child_public_snapshot(
    child: Dict[str, Any], now: Optional[float] = None
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    state = str(child.get("status") or "queued")
    return {
        "child_id": child.get("child_id"),
        "status": state,
        "terminal": state in _REQUIRED_TERMINAL_STATES,
        "elapsed_seconds": round(
            max(
                0.0,
                now - float(
                    child.get("dispatched_at")
                    or child.get("last_meaningful_at")
                    or now
                ),
            ),
            2,
        ),
        "last_liveness_at": child.get("last_liveness_at"),
        "last_meaningful_at": child.get("last_meaningful_at"),
        "last_activity": child.get("last_activity"),
        "current_tool": child.get("current_tool"),
        "in_flight": _child_has_in_flight_source_locked(child),
        "progress_generation": int(child.get("progress_generation") or 0),
    }


def _child_has_in_flight_source_locked(child: Dict[str, Any]) -> bool:
    """True while ANY descendant source still has a call in flight.

    ``in_flight_sources`` is a set keyed by descendant agent instance (see
    ``note_required_progress``'s ``in_flight_source``), not a single bool —
    see the comment on the field's initializer in
    ``dispatch_async_delegation_batch`` for why (nested grandchildren share
    one child slot).
    """
    return bool(child.get("in_flight_sources"))


def _required_child_effective_timeout_locked(
    record: Dict[str, Any], child: Dict[str, Any]
) -> float:
    """Pick the no-progress ceiling that applies to one child right now.

    A child silently inside a tool call (``current_tool`` set) or with a
    provider API call in flight (``in_flight_sources`` non-empty) gets the
    wider in-flight ceiling: both states are legitimate work that produces
    zero liveness touches for the whole duration by design (see
    ``note_required_progress`` / ``note_required_child_activity``), so
    judging them against the tight idle ceiling would kill work that is
    simply slow, not stuck. An idle child (no tool, no call in flight)
    still has to prove progress within the tight ceiling.
    """
    if child.get("current_tool") or _child_has_in_flight_source_locked(child):
        return max(
            0.001,
            float(record.get("in_flight_no_progress_timeout_seconds") or 1500.0),
        )
    return max(0.001, float(record.get("no_progress_timeout_seconds") or 300.0))


def _required_timed_out_child_ids_locked(
    record: Dict[str, Any], now: float
) -> List[str]:
    return [
        str(child_id)
        for child_id, child in (record.get("child_supervision") or {}).items()
        if isinstance(child, dict)
        and child.get("status") not in _REQUIRED_TERMINAL_STATES
        and child.get("status") != "queued"
        and child.get("started_at") is not None
        and now - float(child.get("last_meaningful_at") or now)
        >= _required_child_effective_timeout_locked(record, child)
    ]


def _claim_required_timeout_locked(
    record: Dict[str, Any], now: float
) -> Optional[Dict[str, Any]]:
    """Win the per-child no-progress deadline under the controller lock."""
    if record.get("status") in _REQUIRED_TERMINAL_STATES:
        return None
    _refresh_required_state_locked(record, now)
    no_progress_timed_out_ids = _required_timed_out_child_ids_locked(
        record, now
    )
    children = record.get("child_supervision") or {}
    start_timeout = max(
        0.001, float(record.get("start_timeout_seconds") or 300.0)
    )
    start_timed_out_ids = [
        str(child_id)
        for child_id, child in children.items()
        if isinstance(child, dict)
        and child.get("status") not in _REQUIRED_TERMINAL_STATES
        and child.get("started_at") is None
        and now
        - float(
            child.get("dispatched_at")
            or record.get("dispatched_at")
            or now
        )
        >= start_timeout
    ]
    timed_out_ids = list(
        dict.fromkeys(no_progress_timed_out_ids + start_timed_out_ids)
    )
    finalization_started_at = record.get("finalization_started_at")
    finalization_timeout = max(
        0.001,
        float(
            record.get("finalization_timeout_seconds")
            or record.get("no_progress_timeout_seconds")
            or 300.0
        ),
    )
    if (
        not timed_out_ids
        and finalization_started_at is not None
        and now - float(finalization_started_at) >= finalization_timeout
    ):
        reason = (
            "Required delegation timed out after all children finished "
            f"because batch finalization did not complete within "
            f"{finalization_timeout:g} seconds."
        )
        record["status"] = "timeout"
        record["completed_at"] = now
        record["last_liveness_at"] = now
        record["last_activity"] = reason
        record["current_tool"] = None
        children = record.get("child_supervision") or {}
        record["result"] = {
            "results": [
                _required_child_result_locked(
                    children.get(str(child_id)),
                    child_id=str(child_id),
                )
                for child_id in (record.get("child_ids") or [])
                if isinstance(children.get(str(child_id)), dict)
            ],
            "error": reason,
            "finalization_timeout": True,
            "total_duration_seconds": round(
                now - float(record.get("dispatched_at") or now), 2
            ),
        }
        action = {
            "child_terminalization": None,
            "child_terminal_statuses": {},
            "interrupt": record.get("interrupt_fn"),
            "done_event": record.get("done_event"),
            "reason": reason,
        }
        _clear_required_terminal_callbacks_locked(record)
        return action
    if not timed_out_ids:
        return None

    # Each child may have timed out against a different ceiling (idle 300s
    # vs. the wider in-flight ceiling for a silent tool/API call) — describe
    # per-child ceilings in the reason whenever they differ instead of
    # quoting one number that would be wrong for some of the listed children.
    no_progress_ceilings = {
        child_id: _required_child_effective_timeout_locked(
            record, children.get(child_id) or {}
        )
        for child_id in no_progress_timed_out_ids
    }
    distinct_ceilings = {round(v, 3) for v in no_progress_ceilings.values()}
    if len(distinct_ceilings) <= 1:
        _timeout_desc = f"{next(iter(distinct_ceilings), 300.0):g} seconds"
        _no_progress_child_desc = ", ".join(no_progress_timed_out_ids)
    else:
        _timeout_desc = "their respective ceilings"
        _no_progress_child_desc = ", ".join(
            f"{child_id} ({no_progress_ceilings[child_id]:g}s)"
            for child_id in no_progress_timed_out_ids
        )
    if start_timed_out_ids and not no_progress_timed_out_ids:
        reason = (
            "Required delegation timed out after "
            f"{start_timeout:g} seconds because child(s) did not start: "
            + ", ".join(start_timed_out_ids)
            + "."
        )
    elif start_timed_out_ids:
        reason = (
            "Required delegation timed out: child(s) "
            + ", ".join(start_timed_out_ids)
            + f" did not start within {start_timeout:g} seconds; child(s) "
            + _no_progress_child_desc
            + f" made no meaningful progress for {_timeout_desc}."
        )
    else:
        reason = (
            "Required delegation timed out after "
            f"{_timeout_desc} without meaningful progress in child(s): "
            + _no_progress_child_desc
            + "."
        )
    child_results = []
    supervision_terminal_ids: List[str] = []
    for child_id in (record.get("child_ids") or []):
        child = children.get(str(child_id))
        if not isinstance(child, dict):
            continue
        if child.get("status") not in _REQUIRED_TERMINAL_STATES:
            supervision_terminal_ids.append(str(child_id))
            # Capture pre-timeout diagnostics BEFORE any field below is
            # overwritten with the generic timeout reason — otherwise the
            # last thing the child was actually doing is lost forever, which
            # is exactly what a caller needs to tell "genuinely stuck" apart
            # from "timed out mid useful work" after the fact.
            pre_timeout_diagnostics = {
                "last_activity": child.get("last_activity"),
                "current_tool": child.get("current_tool"),
                "last_meaningful_at": child.get("last_meaningful_at"),
            }
            logger.warning(
                "Required delegation child %s timed out (status=%s, "
                "current_tool=%r, last_activity=%r, last_meaningful_at=%s)",
                child_id,
                child.get("status"),
                pre_timeout_diagnostics["current_tool"],
                pre_timeout_diagnostics["last_activity"],
                pre_timeout_diagnostics["last_meaningful_at"],
            )
            child["status"] = (
                "timeout" if str(child_id) in timed_out_ids else "interrupted"
            )
            child["completed_at"] = now
            child["last_liveness_at"] = now
            child["last_activity"] = reason
            child["current_tool"] = None
            child["terminal_source"] = "controller"
            child["result"] = {
                "child_id": str(child_id),
                "status": child.get("status"),
                "summary": None,
                "error": (
                    reason if str(child_id) in timed_out_ids else None
                ),
                "diagnostics": pre_timeout_diagnostics,
            }
        child_results.append(
            _required_child_result_locked(
                child,
                child_id=str(child_id),
                reason=reason if str(child_id) in timed_out_ids else None,
            )
        )

    record["status"] = "timeout"
    record["completed_at"] = now
    record["last_liveness_at"] = now
    record["last_activity"] = reason
    record["current_tool"] = None
    record["result"] = {
        "results": child_results,
        "error": reason,
        "timed_out_child_ids": timed_out_ids,
        "total_duration_seconds": round(
            now - float(record.get("dispatched_at") or now), 2
        ),
    }
    terminalization = _claim_required_child_terminalization_locked(
        record, supervision_terminal_ids
    )
    action = {
        "child_terminalization": terminalization,
        "child_terminal_statuses": {
            child_id: str(children[child_id].get("status") or "failed")
            for child_id in supervision_terminal_ids
            if isinstance(children.get(child_id), dict)
        },
        "interrupt": record.get("interrupt_fn"),
        "done_event": record.get("done_event"),
        "reason": reason,
    }
    _clear_required_terminal_callbacks_locked(record)
    return action


def _finish_required_timeout(action: Optional[Dict[str, Any]]) -> None:
    """Publish terminal child frames, interrupt work, then open the wait gate."""
    if action is None:
        return
    _emit_required_child_terminalization(
        action.get("child_terminalization"),
        status=action.get("child_terminal_statuses") or "timeout",
        reason=str(action.get("reason") or "Required delegation timed out."),
    )
    try:
        interrupt = action.get("interrupt")
        if callable(interrupt):
            interrupt()
    except Exception:
        logger.debug("Timed-out required delegation interrupt failed", exc_info=True)
    finally:
        done_event = action.get("done_event")
        if isinstance(done_event, threading.Event):
            done_event.set()


def _supervise_required_delegation(
    delegation_id: str,
) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not record.get("required"):
            return None
        action = _claim_required_timeout_locked(record, now)
        snapshot = _required_public_snapshot(record)
    _finish_required_timeout(action)
    return snapshot


def _required_watchdog(delegation_id: str) -> None:
    """Enforce the deadline even when the parent never calls wait/status.

    This daemon thread is the ONLY deadline enforcement for a required
    delegation the parent never polls via wait_required/required_status. An
    unhandled exception here must never end the loop silently — that would
    leave the delegation running forever with no one left to terminalize it.
    Every iteration is guarded; on error, log loudly and keep polling
    instead of dying (mirrors the "must never crash the worker" guard on
    the batch _worker above).
    """
    while True:
        try:
            with _records_lock:
                record = _records.get(str(delegation_id or ""))
                if record is None or not record.get("required"):
                    return
                done_event = record.get("done_event")
                timeout = max(
                    0.001,
                    float(record.get("no_progress_timeout_seconds") or 300.0),
                )
                start_timeout = max(
                    0.001,
                    float(record.get("start_timeout_seconds") or 300.0),
                )
            interval = min(
                5.0, max(0.01, min(timeout, start_timeout) / 10.0)
            )
            if isinstance(done_event, threading.Event) and done_event.wait(interval):
                return
            snapshot = _supervise_required_delegation(delegation_id)
            if snapshot is None or snapshot.get("terminal"):
                return
        except Exception:  # noqa: BLE001 — must never crash the watchdog
            logger.exception(
                "Required delegation watchdog for %s hit an unexpected "
                "error; continuing to enforce the deadline.",
                delegation_id,
            )
            # Guard against a tight busy-loop if the error recurs on every
            # iteration (e.g. a persistently malformed record).
            time.sleep(1.0)


def required_status(parent_agent, delegation_id: str) -> Dict[str, Any]:
    """Read one required delegation without consuming its terminal result."""
    _supervise_required_delegation(delegation_id)
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not _required_owner_matches(record, parent_agent):
            return {"status": "not_found", "delegation_id": delegation_id}
        payload = _required_public_snapshot(record)
        if payload["terminal"]:
            payload["result"] = record.get("result")
        return payload


def has_unconsumed_required(parent_agent) -> bool:
    """True while the current ACP turn owns any unconsumed required result."""
    with _records_lock:
        return any(
            _required_owner_matches(record, parent_agent)
            and record.get("consumed_at") is None
            for record in _records.values()
        )


def list_unconsumed_required(parent_agent) -> List[Dict[str, Any]]:
    """Ordered same-turn required records, including terminal results."""
    with _records_lock:
        records = [
            record
            for record in _records.values()
            if _required_owner_matches(record, parent_agent)
            and record.get("consumed_at") is None
        ]
        records.sort(
            key=lambda record: (
                float(record.get("dispatched_at") or 0),
                str(record.get("delegation_id") or ""),
            )
        )
        return [_required_public_snapshot(record) for record in records]


def wait_required(
    parent_agent,
    delegation_id: str,
    *,
    timeout_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Bounded wait without consuming the terminal result."""
    _supervise_required_delegation(delegation_id)
    record = _required_record(parent_agent, delegation_id)
    if record is None:
        return {"status": "not_found", "delegation_id": delegation_id}
    done_event = record.get("done_event")
    if isinstance(done_event, threading.Event):
        done_event.wait(max(0.0, min(float(timeout_seconds or 0.0), 30.0)))
    _supervise_required_delegation(delegation_id)
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not _required_owner_matches(record, parent_agent):
            return {"status": "not_found", "delegation_id": delegation_id}
        payload = _required_public_snapshot(record)
        if payload["terminal"]:
            payload["result"] = record.get("result")
        return payload


def observe_required(
    parent_agent,
    delegation_id: str,
    append_observation: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Append a terminal observation and open the gate — a two-phase consume.

    ``append_observation`` does a real durable write (SQLite via
    hermes_state: BEGIN IMMEDIATE with up to 15 retries, worst case ~2s+).
    It must never run while holding ``_records_lock`` — that lock is the
    single choke point for every OTHER session's heartbeats, watchdogs,
    status, and cancel calls, and a slow write would stall all of them.

    So this is three phases instead of one locked action:

    1. (locked) Verify the record is terminal and unconsumed, then win an
       exclusive ``consuming`` claim. A concurrent caller for the same
       record sees the claim and returns without appending a second
       observation — consumed exactly once.
    2. (UNLOCKED) Run ``append_observation``.
    3. (locked) On success: stamp ``consumed_at``, clear the claim, clear
       terminal callbacks, prune. On exception: clear the claim only
       (leave ``consumed_at`` unset) so the record stays consumable by a
       retry, then propagate — the gate must never read as open for an
       observation that was never durably persisted.

    While a claim is in flight, ``consumed_at`` stays ``None`` so
    ``has_unconsumed_required``/``list_unconsumed_required`` keep reporting
    the record as pending, and ``stop_required_for_agent`` skips it instead
    of clearing callbacks or stamping ``consumed_at`` out from under the
    in-flight persist (which would let STOP race ahead of durable storage
    and double-terminalize the record's consumption state).
    """
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not _required_owner_matches(record, parent_agent):
            return {"status": "not_found", "delegation_id": delegation_id}
        payload = _required_public_snapshot(record)
        if not payload["terminal"]:
            return payload
        payload["result"] = record.get("result")
        if record.get("consumed_at") is not None:
            # Already durably consumed by an earlier winner.
            _clear_required_terminal_callbacks_locked(record)
            _prune_completed_locked()
            payload["consumed"] = True
            return payload
        if record.get("consuming"):
            # Another caller already won the claim and is persisting (or
            # will retry after a failure). Don't append a second
            # observation or touch consumed_at/callbacks ourselves.
            payload["consumed"] = False
            return payload
        record["consuming"] = True

    succeeded = False
    try:
        append_observation(dict(payload))
        succeeded = True
    finally:
        # `finally`, not `except Exception`: append_observation's real
        # durable write (SQLite BEGIN IMMEDIATE) can in principle raise a
        # BaseException that is NOT an Exception subclass (SystemExit/
        # KeyboardInterrupt delivered to this thread mid-write). An
        # `except Exception` clause would skip those and leak the claim
        # forever — has_unconsumed_required would report the record
        # pending forever, stop_required_for_agent would skip it forever
        # (its "and not record.get('consuming')" guard), and
        # _prune_completed_locked can never evict it either. `succeeded`
        # preserves the existing success/failure semantics: only clear the
        # claim WITHOUT stamping consumed_at when append_observation did
        # NOT return normally (any exception type) — on success, phase 3
        # below still owns setting consumed_at.
        if not succeeded:
            with _records_lock:
                record = _records.get(str(delegation_id or ""))
                if record is not None:
                    record["consuming"] = False

    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is not None:
            record["consumed_at"] = time.time()
            record["consuming"] = False
            _clear_required_terminal_callbacks_locked(record)
            _prune_completed_locked()
    payload["consumed"] = True
    return payload


def cancel_required(
    parent_agent,
    delegation_id: str,
    *,
    child_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Lock-win cancellation that never waits for a child to cooperate.

    A whole-batch cancel is terminal immediately. A targeted cancel marks that
    child terminal immediately and leaves the controller open only while other
    children are still live. All callbacks run after releasing the registry
    lock; late child completions cannot overwrite the winning terminal state.
    """
    now = time.time()
    action: Optional[Dict[str, Any]] = None
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not _required_owner_matches(record, parent_agent):
            return {"status": "not_found", "delegation_id": delegation_id}
        child_ids = list(record.get("child_ids") or [])
        if child_id is not None and child_id not in child_ids:
            return {"status": "not_found", "delegation_id": delegation_id}
        if record.get("status") in _REQUIRED_TERMINAL_STATES:
            payload = _required_public_snapshot(record)
            payload["result"] = record.get("result")
            return payload

        children = record.get("child_supervision") or {}
        if child_id is not None:
            cancelled = record.setdefault("cancelled_child_ids", [])
            child = children.get(str(child_id))
            if not isinstance(child, dict):
                return {"status": "not_found", "delegation_id": delegation_id}
            if child.get("status") in _REQUIRED_TERMINAL_STATES:
                return _required_public_snapshot(record)
            if child_id not in cancelled:
                cancelled.append(child_id)
            reason = f"Required delegation child {child_id} was cancelled."
            child["status"] = "cancelled"
            child["completed_at"] = now
            child["last_liveness_at"] = now
            child["last_activity"] = reason
            child["current_tool"] = None
            child["terminal_source"] = "controller"
            child["result"] = {
                "child_id": str(child_id),
                "status": "cancelled",
                "summary": None,
                "error": reason,
            }

            terminalization = _claim_required_child_terminalization_locked(
                record, [str(child_id)]
            )
            remaining = [
                candidate
                for candidate in children.values()
                if isinstance(candidate, dict)
                and candidate.get("status") not in _REQUIRED_TERMINAL_STATES
            ]
            all_terminal = not remaining
            if all_terminal:
                record["status"] = "cancelled"
                record["completed_at"] = now
                record["last_liveness_at"] = now
                record["last_activity"] = reason
                record["current_tool"] = None
                record["result"] = {
                    "results": [
                        _required_child_result_locked(
                            children.get(str(candidate_id)),
                            child_id=str(candidate_id),
                        )
                        for candidate_id in child_ids
                        if isinstance(children.get(str(candidate_id)), dict)
                    ],
                    "error": reason,
                    "cancelled_child_ids": list(cancelled),
                    "total_duration_seconds": round(
                        now - float(record.get("dispatched_at") or now), 2
                    ),
                }
            else:
                _refresh_required_state_locked(record, now)
            action = {
                "child_terminalization": terminalization,
                "child_interrupt": record.get("child_interrupt_fn"),
                "batch_interrupt": (
                    record.get("interrupt_fn") if all_terminal else None
                ),
                "done_event": record.get("done_event") if all_terminal else None,
                "child_id": str(child_id),
                "reason": reason,
            }
            if all_terminal:
                _clear_required_terminal_callbacks_locked(record)
        else:
            record["cancel_requested"] = True
            reason = "Required delegation was cancelled."
            cancelled = record.setdefault("cancelled_child_ids", [])
            newly_cancelled: List[str] = []
            for candidate_id in child_ids:
                child = children.get(str(candidate_id))
                if (
                    not isinstance(child, dict)
                    or child.get("status") in _REQUIRED_TERMINAL_STATES
                ):
                    continue
                child["status"] = "cancelled"
                child["completed_at"] = now
                child["last_liveness_at"] = now
                child["last_activity"] = reason
                child["current_tool"] = None
                child["terminal_source"] = "controller"
                child["result"] = {
                    "child_id": str(candidate_id),
                    "status": "cancelled",
                    "summary": None,
                    "error": reason,
                }
                newly_cancelled.append(str(candidate_id))
                if candidate_id not in cancelled:
                    cancelled.append(candidate_id)

            record["status"] = "cancelled"
            record["completed_at"] = now
            record["last_liveness_at"] = now
            record["last_activity"] = reason
            record["current_tool"] = None
            record["result"] = {
                "results": [
                    _required_child_result_locked(
                        children.get(str(candidate_id)),
                        child_id=str(candidate_id),
                    )
                    for candidate_id in child_ids
                    if isinstance(children.get(str(candidate_id)), dict)
                ],
                "error": reason,
                "cancelled_child_ids": list(cancelled),
                "total_duration_seconds": round(
                    now - float(record.get("dispatched_at") or now), 2
                ),
            }
            action = {
                "child_terminalization": (
                    _claim_required_child_terminalization_locked(
                        record, newly_cancelled
                    )
                ),
                "batch_interrupt": record.get("interrupt_fn"),
                "done_event": record.get("done_event"),
                "reason": reason,
            }
            _clear_required_terminal_callbacks_locked(record)

        payload = _required_public_snapshot(record)
        if payload["terminal"]:
            payload["result"] = record.get("result")

    _emit_required_child_terminalization(
        action.get("child_terminalization") if action else None,
        status="cancelled",
        reason=str(
            (action or {}).get("reason")
            or "Required delegation was cancelled."
        ),
    )
    child_interrupt = (action or {}).get("child_interrupt")
    if callable(child_interrupt):
        try:
            child_interrupt(str((action or {}).get("child_id") or ""))
        except Exception:
            logger.debug(
                "Required child cancellation interrupt failed", exc_info=True
            )
    batch_interrupt = (action or {}).get("batch_interrupt")
    if callable(batch_interrupt):
        try:
            batch_interrupt()
        except Exception:
            logger.debug(
                "Required batch cancellation interrupt failed", exc_info=True
            )
    done_event = (action or {}).get("done_event")
    if isinstance(done_event, threading.Event):
        done_event.set()
    return payload


def interrupt_required_for_agent(parent_agent, reason: str = "parent interrupted") -> int:
    """Cancel every unconsumed required record owned by the current turn."""
    with _records_lock:
        ids = [
            str(record.get("delegation_id") or "")
            for record in _records.values()
            if _required_owner_matches(record, parent_agent)
            and record.get("status") not in _REQUIRED_TERMINAL_STATES
        ]
    for delegation_id in ids:
        cancel_required(parent_agent, delegation_id)
    if ids:
        logger.info("Interrupted %d required delegation(s): %s", len(ids), reason)
    return len(ids)


def stop_required_for_agent(parent_agent, reason: str = "parent stopped") -> int:
    """Terminalize owned records without fabricating model observations.

    Records with an in-flight ``observe_required`` consuming claim
    (``consuming`` True) are skipped: that claim owns a durable write in
    progress outside the lock (see ``observe_required``), and stamping
    ``consumed_at``/clearing callbacks here would race ahead of it — STOP
    would appear to have terminalized consumption before the observation
    was ever persisted. The in-flight consume finishes (or fails and
    clears its own claim) on its own; this call simply leaves it alone.
    """
    terminalizations = []
    now = time.time()
    with _records_lock:
        records = [
            record for record in _records.values()
            if _required_owner_matches(record, parent_agent)
            and record.get("consumed_at") is None
            and not record.get("consuming")
        ]
        for record in records:
            if record.get("status") not in _REQUIRED_TERMINAL_STATES:
                record["status"] = "cancelled"
                record["completed_at"] = now
                record["cancel_requested"] = True
                child_results = []
                supervision_terminal_ids: List[str] = []
                children = record.get("child_supervision") or {}
                for child_id in (record.get("child_ids") or []):
                    child = children.get(str(child_id))
                    if not isinstance(child, dict):
                        continue
                    if child.get("status") not in _REQUIRED_TERMINAL_STATES:
                        supervision_terminal_ids.append(str(child_id))
                        child["status"] = "cancelled"
                        child["completed_at"] = now
                        child["last_liveness_at"] = now
                        child["last_activity"] = reason
                        child["current_tool"] = None
                        child["terminal_source"] = "controller"
                        child["result"] = {
                            "child_id": str(child_id),
                            "status": "cancelled",
                            "summary": None,
                            "error": reason,
                        }
                    child_results.append(
                        _required_child_result_locked(
                            child,
                            child_id=str(child_id),
                            reason=(
                                reason
                                if child.get("status") == "cancelled"
                                else None
                            ),
                        )
                    )
                record["result"] = {
                    "results": child_results,
                    "error": reason,
                    "total_duration_seconds": round(
                        now - float(record.get("dispatched_at") or now), 2
                    ),
                }
                terminalizations.append(
                    (
                        _claim_required_child_terminalization_locked(
                            record, supervision_terminal_ids
                        ),
                        record.get("interrupt_fn"),
                        record.get("done_event"),
                    )
                )
                _clear_required_terminal_callbacks_locked(record)
            elif record.get("status") in _REQUIRED_TERMINAL_STATES:
                _clear_required_terminal_callbacks_locked(record)
            record["consumed_at"] = now
        _prune_completed_locked()
    for child_terminalization, interrupt, done_event in terminalizations:
        _emit_required_child_terminalization(
            child_terminalization,
            status="cancelled",
            reason=reason,
        )
        try:
            if callable(interrupt):
                interrupt()
        except Exception:
            logger.debug("Stopped required delegation interrupt failed", exc_info=True)
        finally:
            if isinstance(done_event, threading.Event):
                done_event.set()
    return len(records)


def _required_child_result_locked(
    child: Optional[Dict[str, Any]],
    *,
    child_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the best canonical child evidence available under the lock."""
    if not isinstance(child, dict):
        return {
            "child_id": str(child_id),
            "status": "failed",
            "summary": None,
            "error": reason or "Required child state was unavailable.",
        }
    stored = child.get("result")
    if isinstance(stored, dict):
        result = dict(stored)
    else:
        result = {}
    result.setdefault("child_id", str(child_id))
    if child.get("task_index") is not None:
        result.setdefault("task_index", child.get("task_index"))
    result.setdefault("status", str(child.get("status") or "failed"))
    if "summary" not in result:
        result["summary"] = None
    if reason and not result.get("error"):
        result["error"] = reason
    return result


def _claim_required_child_terminalization_locked(
    record: Dict[str, Any],
    child_ids: Optional[List[str]] = None,
) -> Optional[tuple[Callable[[str, str, str], None], List[str]]]:
    """Claim selected not-yet-supervision-terminalized children exactly once.

    Caller holds ``_records_lock``. The returned callback is invoked only
    after releasing the controller lock: ACP delivery can cross threads and
    must never block registry state transitions.
    """
    callback = record.get("child_terminal_fn")
    if not callable(callback):
        return None
    already = {
        str(child_id)
        for child_id in (record.get("terminalized_child_ids") or [])
    }
    selected = (
        [str(child_id) for child_id in child_ids]
        if child_ids is not None
        else [str(child_id) for child_id in (record.get("child_ids") or [])]
    )
    pending = [
        str(child_id)
        for child_id in selected
        if str(child_id) and str(child_id) not in already
    ]
    if not pending:
        return None
    record.setdefault("terminalized_child_ids", []).extend(pending)
    all_ids = {
        str(child_id)
        for child_id in (record.get("child_ids") or [])
        if str(child_id)
    }
    claimed_ids = {
        str(child_id)
        for child_id in (record.get("terminalized_child_ids") or [])
    }
    if all_ids <= claimed_ids:
        record["child_terminal_fn"] = None
    return callback, pending


def _emit_required_child_terminalization(
    claimed: Optional[tuple[Callable[[str, str, str], None], List[str]]],
    *,
    status: str | Dict[str, str],
    reason: str,
) -> None:
    """Close claimed child UI frames; late real completions become no-ops."""
    if claimed is None:
        return
    callback, child_ids = claimed
    for child_id in child_ids:
        try:
            child_status = (
                str(status.get(child_id) or "failed")
                if isinstance(status, dict)
                else status
            )
            callback(child_id, child_status, reason)
        except Exception:
            logger.debug(
                "Required child supervision terminal update failed",
                exc_info=True,
            )


def note_required_progress(
    delegation_id: str,
    *,
    child_id: str,
    current_tool: Optional[str],
    activity: str,
    meaningful: bool,
    state: Optional[str] = None,
    in_flight: Optional[bool] = None,
    in_flight_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update required liveness/progress clocks without changing ownership.

    ``in_flight`` is an explicit state marker (never inferred from
    ``meaningful``) for "this child has a provider API call outstanding
    right now". Pass ``True`` when dispatching a request and ``False`` once
    it returns; omit (``None``) on touches that don't cross that boundary so
    the previously recorded state is preserved through a silent wait. See
    ``_required_child_effective_timeout_locked``.

    ``in_flight_source`` identifies WHICH descendant agent instance is
    setting/clearing the marker (typically its own ``_subagent_id``).
    Nested required delegation lets several grandchildren share one child
    slot (they inherit the same frozen
    ``_required_delegation_ancestor_binding``), so in-flight state is
    tracked per source in ``child["in_flight_sources"]`` rather than as one
    shared bool — a sibling finishing its call must not clear the marker
    while a DIFFERENT sibling is still genuinely in flight. Falls back to
    ``child_id`` itself when omitted (correct for a direct child touching
    its own slot, where source and child_id are the same agent).
    """
    now = time.time()
    action = None
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not record.get("required"):
            return None
        if record.get("status") in _REQUIRED_TERMINAL_STATES:
            return _required_public_snapshot(record)
        children = record.get("child_supervision") or {}
        child = children.get(str(child_id or ""))
        if not isinstance(child, dict):
            return None
        if child.get("status") in _REQUIRED_TERMINAL_STATES:
            return _required_child_public_snapshot(child, now)
        child["last_liveness_at"] = now
        child["last_activity"] = str(activity or "")
        child["current_tool"] = current_tool
        if in_flight is not None:
            source = str(in_flight_source or child_id or "")
            sources = child.setdefault("in_flight_sources", set())
            if in_flight:
                sources.add(source)
            else:
                sources.discard(source)
        if child.get("status") == "queued" and (
            state == "running" or meaningful
        ):
            child["started_at"] = now
            child["last_meaningful_at"] = now
            child["status"] = "running"
        if meaningful:
            child["last_meaningful_at"] = now
            child["progress_generation"] = (
                int(child.get("progress_generation") or 0) + 1
            )
            child["status"] = "running"
        elif state in {"running", "slow", "stalled"}:
            child["status"] = state
        _refresh_required_state_locked(record, now)
        action = _claim_required_timeout_locked(record, now)
        snapshot = _required_child_public_snapshot(child, now)
    _finish_required_timeout(action)
    return snapshot


def note_required_child_terminal(
    child_id: str,
    *,
    status: str,
    activity: str,
    result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Mark one child terminal without finalizing its batch result."""
    now = time.time()
    normalized = str(status or "completed").strip().lower()
    if normalized in {"complete", "success", "done", ""}:
        normalized = "completed"
    elif normalized not in _REQUIRED_TERMINAL_STATES:
        normalized = "failed"
    with _records_lock:
        for record in _records.values():
            if not record.get("required"):
                continue
            child = (record.get("child_supervision") or {}).get(str(child_id))
            if not isinstance(child, dict):
                continue
            # A controller timeout/cancel may have won under the lock while
            # its ACP terminal callback is still queued outside the lock.
            # Return that winner to a racing normal completion so the relay
            # can emit the controller-owned status instead of lying
            # "completed". Terminal child state is immutable.
            if child.get("status") in _REQUIRED_TERMINAL_STATES:
                return _required_child_public_snapshot(child, now)
            if record.get("status") in _REQUIRED_TERMINAL_STATES:
                return _required_child_public_snapshot(child, now)
            if child.get("status") not in _REQUIRED_TERMINAL_STATES:
                child["status"] = normalized
                child["completed_at"] = now
                child["last_liveness_at"] = now
                child["last_activity"] = str(activity or normalized)
                child["current_tool"] = None
                child["terminal_source"] = "worker"
                if isinstance(result, dict):
                    stored = dict(result)
                    stored.setdefault("child_id", str(child_id))
                    stored.setdefault("status", normalized)
                    child["result"] = stored
            _refresh_required_state_locked(record, now)
            return _required_child_public_snapshot(child, now)
    return None


def note_required_child_activity(
    child_id: str,
    *,
    current_tool: Optional[str],
    activity: str,
    meaningful: bool,
    state: Optional[str] = None,
    in_flight: Optional[bool] = None,
    in_flight_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update the active required record that owns ``child_id``."""
    with _records_lock:
        delegation_id = next(
            (
                str(record.get("delegation_id") or "")
                for record in _records.values()
                if record.get("required")
                and child_id in (record.get("child_ids") or [])
                and record.get("status") not in _REQUIRED_TERMINAL_STATES
            ),
            "",
        )
    if not delegation_id:
        return None
    return note_required_progress(
        delegation_id,
        child_id=child_id,
        current_tool=current_tool,
        activity=activity,
        meaningful=meaningful,
        state=state,
        in_flight=in_flight,
        in_flight_source=in_flight_source,
    )


def _refresh_required_state_locked(
    record: Dict[str, Any], now: Optional[float] = None
) -> None:
    if record.get("status") in _REQUIRED_TERMINAL_STATES:
        return
    now = time.time() if now is None else now
    children = record.get("child_supervision") or {}
    active_children = []
    queued_children = []
    for child in children.values():
        if (
            not isinstance(child, dict)
            or child.get("status") in _REQUIRED_TERMINAL_STATES
        ):
            continue
        if (
            child.get("status") == "queued"
            and child.get("started_at") is None
        ):
            queued_children.append(child)
            continue
        active_children.append(child)
        age = max(
            0.0, now - float(child.get("last_meaningful_at") or now)
        )
        # Bucket against THIS child's ceiling (idle vs. in-flight) so a
        # long-running tool/API call isn't shown "stalled" purely because it
        # crossed 80% of the tight idle threshold.
        child_timeout = _required_child_effective_timeout_locked(record, child)
        if age >= child_timeout * 0.8:
            child["status"] = "stalled"
        elif age >= child_timeout * 0.5:
            child["status"] = "slow"
        else:
            child["status"] = "running"

    if active_children:
        states = {str(child.get("status") or "") for child in active_children}
        if "stalled" in states:
            record["status"] = "stalled"
        elif "slow" in states:
            record["status"] = "slow"
        else:
            record["status"] = "running"
        record["last_liveness_at"] = max(
            float(child.get("last_liveness_at") or 0.0)
            for child in active_children
        )
        # The oldest unresolved child's clock governs the batch deadline; a
        # productive sibling can never mask a stuck child.
        record["last_meaningful_at"] = min(
            float(child.get("last_meaningful_at") or now)
            for child in active_children
        )
        newest = max(
            active_children,
            key=lambda child: float(child.get("last_liveness_at") or 0.0),
        )
        record["last_activity"] = newest.get("last_activity")
        record["current_tool"] = newest.get("current_tool")
        record["progress_generation"] = sum(
            int(child.get("progress_generation") or 0)
            for child in children.values()
            if isinstance(child, dict)
        )
    elif queued_children:
        record["status"] = "queued"
        record["last_activity"] = "queued"
        record["current_tool"] = None
        record["finalization_started_at"] = None
    elif children:
        # Child terminal callbacks are emitted before _run_single_child's
        # cleanup/future aggregation finishes. Bound that finalization phase
        # independently so a wedged heartbeat join, child.close(), hook, or
        # aggregator cannot leave the parent waiting forever after every child
        # card already says terminal.
        if record.get("finalization_started_at") is None:
            record["finalization_started_at"] = now
        record["status"] = "finalizing"
        record["last_activity"] = "finalizing required delegation"
        record["current_tool"] = None


def refresh_required_supervision(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Derive running/slow/stalled from the meaningful-progress clock."""
    return _supervise_required_delegation(delegation_id)


def _dispatch_required_batch(
    *, goals: List[str], context: Optional[str], toolsets: Optional[List[str]], role: str,
    model: Optional[str], session_key: str, parent_session_id: Optional[str],
    runner: Callable[[], Dict[str, Any]], parent_owner_token: str,
    origin_ui_session_id: str, origin_session_id: str,
    interrupt_fn: Optional[Callable[[], None]], max_async_children: int,
    delegation_id: Optional[str], progress_fn: Optional[Callable[[], tuple]],
    parent_turn_id: str, child_ids: Optional[List[str]],
    child_interrupt_fn: Optional[Callable[[str], None]],
    child_terminal_fn: Optional[Callable[[str, str, str], None]],
    no_progress_timeout_seconds: float, start_timeout_seconds: Optional[float],
    in_flight_no_progress_timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    """Dispatch the ACP controller batch without entering the legacy queue."""
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    required_child_ids = [str(child_id) for child_id in (child_ids or []) if str(child_id)]
    n = len(goals)
    combined_goal = goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    record: Dict[str, Any] = {
        "delegation_id": delegation_id, "goal": combined_goal, "goals": list(goals),
        "context": context, "toolsets": list(toolsets) if toolsets else None,
        "role": role, "model": model, "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id, "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id, "parent_owner_token": str(parent_owner_token or ""),
        "parent_turn_id": str(parent_turn_id or ""), "required": True,
        "status": "queued", "dispatched_at": dispatched_at, "completed_at": None,
        "interrupt_fn": interrupt_fn, "child_interrupt_fn": child_interrupt_fn,
        "child_terminal_fn": child_terminal_fn, "done_event": threading.Event(),
        "is_batch": True, "progress_fn": progress_fn, "_progress_token": None,
        "_progress_ts": dispatched_at, "_interrupted_at": None,
        "child_ids": required_child_ids,
        "child_supervision": {
            child_id: {
                "child_id": child_id, "task_index": task_index, "status": "queued",
                "dispatched_at": dispatched_at, "started_at": None,
                "last_liveness_at": dispatched_at, "last_meaningful_at": dispatched_at,
                "last_activity": "queued", "current_tool": None,
                "in_flight_sources": set(), "progress_generation": 0,
                "completed_at": None, "result": None,
            }
            for task_index, child_id in enumerate(required_child_ids)
        },
        "terminalized_child_ids": [], "cancel_requested": False,
        "cancelled_child_ids": [], "consumed_at": None, "consuming": False,
        "result": None, "last_liveness_at": dispatched_at,
        "last_meaningful_at": dispatched_at, "last_activity": "queued",
        "current_tool": None, "progress_generation": 0,
        "finalization_started_at": None,
        "no_progress_timeout_seconds": max(0.001, float(no_progress_timeout_seconds or 300.0)),
        "finalization_timeout_seconds": max(0.001, float(no_progress_timeout_seconds or 300.0)),
        "start_timeout_seconds": max(
            0.001, float(start_timeout_seconds if start_timeout_seconds is not None else (no_progress_timeout_seconds or 300.0))
        ),
        "in_flight_no_progress_timeout_seconds": max(
            0.001, float(in_flight_no_progress_timeout_seconds if in_flight_no_progress_timeout_seconds is not None else 1500.0)
        ),
    }
    with _records_lock:
        running = sum(1 for item in _records.values() if item.get("status") in _LIVE_DELEGATION_STATES)
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} running). Wait for one to finish "
                    "or raise delegation.max_concurrent_children in config.yaml to allow more concurrent units."
                ),
            }
        _records[delegation_id] = record
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            with _records_lock:
                live = _records.get(delegation_id)
                if live is not None and live.get("status") == "queued":
                    live["status"] = "running"
                    live["last_activity"] = "running"
                    live["last_liveness_at"] = time.time()
            combined = runner() or {}
            child_results = combined.get("results") or []
            status = "error" if child_results and all(
                item.get("status") not in ("completed", "success") for item in child_results if isinstance(item, dict)
            ) else "completed"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Required async delegation batch %s crashed", delegation_id)
            combined = {"results": [], "error": f"{type(exc).__name__}: {exc}", "total_duration_seconds": round(time.time() - dispatched_at, 2)}
            status = "error"
        finally:
            _finalize_batch(delegation_id, combined, status)

    try:
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        return {"status": "rejected", "error": f"Failed to schedule required async delegation batch: {exc}"}
    if progress_fn is not None:
        _ensure_stale_monitor()
    threading.Thread(
        target=_required_watchdog, args=(delegation_id,), daemon=True,
        name=f"required-watchdog-{delegation_id}",
    ).start()
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(delegation_id: str, combined: Dict[str, Any], status: str) -> None:
    """Terminalize a required batch from worker evidence, preserving controller wins."""
    terminalization = None
    interrupt = None
    done_event = None
    with _records_lock:
        record = _records.get(str(delegation_id or ""))
        if record is None or not record.get("required"):
            return
        if record.get("status") in _REQUIRED_TERMINAL_STATES:
            return
        children = record.get("child_supervision") or {}
        child_ids = [str(value) for value in (record.get("child_ids") or [])]
        incoming = [item for item in (combined.get("results") or []) if isinstance(item, dict)]
        by_id = {}
        for index, item in enumerate(incoming):
            child_id = str(item.get("child_id") or "")
            if not child_id:
                task_index = item.get("task_index")
                if isinstance(task_index, int) and 0 <= task_index < len(child_ids):
                    child_id = child_ids[task_index]
                elif index < len(child_ids):
                    child_id = child_ids[index]
            if child_id:
                by_id[child_id] = item
        unresolved = []
        now = time.time()
        for child_id in child_ids:
            child = children.get(child_id)
            if not isinstance(child, dict):
                continue
            controller_owned = child.get("terminal_source") == "controller" or child_id in {
                str(value) for value in (record.get("terminalized_child_ids") or [])
            }
            item = by_id.get(child_id)
            if item is not None and not controller_owned:
                child_status = str(item.get("status") or "completed").strip().lower()
                if child_status in {"complete", "success", "done", ""}:
                    child_status = "completed"
                elif child_status not in _REQUIRED_TERMINAL_STATES:
                    child_status = "failed"
                child.update(
                    status=child_status, completed_at=now, last_liveness_at=now,
                    last_activity=child_status, current_tool=None,
                    terminal_source="worker", result={**item, "child_id": child_id, "status": child_status},
                )
            if child.get("status") not in _REQUIRED_TERMINAL_STATES:
                unresolved.append(child_id)
        runner_failed = str(status or "").lower() not in {"completed", "complete", "success", "done"} or bool(unresolved)
        reason = str(combined.get("error") or "Required delegation batch ended before every child produced a terminal result.")
        if runner_failed:
            for child_id in unresolved:
                child = children.get(child_id)
                if not isinstance(child, dict):
                    continue
                child.update(
                    status="failed", completed_at=now, last_liveness_at=now,
                    last_activity=reason, current_tool=None, terminal_source="controller",
                    result={"child_id": child_id, "status": "failed", "summary": None, "error": reason},
                )
        canonical = (
            [
                _required_child_result_locked(
                    children.get(child_id), child_id=child_id
                )
                for child_id in child_ids
                if isinstance(children.get(child_id), dict)
            ]
            if child_ids
            else incoming
        )
        aggregate = dict(combined)
        aggregate["results"] = canonical
        if runner_failed:
            aggregate["error"] = reason
        child_states = {str(item.get("status") or "") for item in canonical}
        if record.get("cancel_requested"):
            final_status = "cancelled"
        elif runner_failed:
            final_status = "failed"
        elif child_states and child_states <= {"cancelled", "interrupted"}:
            final_status = "cancelled"
        elif "timeout" in child_states:
            final_status = "timeout"
        else:
            final_status = str(status or "completed")
        record.update(
            status=final_status, completed_at=now, last_liveness_at=now,
            last_activity=final_status, current_tool=None, result=aggregate, progress_fn=None,
        )
        done_event = record.get("done_event")
        interrupt = record.get("interrupt_fn") if runner_failed and unresolved else None
        terminalization = _claim_required_child_terminalization_locked(record, unresolved) if runner_failed and unresolved else None
        _clear_required_terminal_callbacks_locked(record)
        _prune_completed_locked()
    _emit_required_child_terminalization(terminalization, status={child_id: "failed" for child_id in unresolved}, reason=reason)
    if callable(interrupt):
        try:
            interrupt()
        except Exception:
            logger.debug("Required batch interrupt failed", exc_info=True)
    if isinstance(done_event, threading.Event):
        done_event.set()


def running_for_session(session_key: str, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
    """Return non-required live records for the ACP join/reaper path."""
    with _records_lock:
        return [
            {
                key: value for key, value in record.items()
                if key not in {"interrupt_fn", "child_interrupt_fn", "child_terminal_fn", "done_event"}
            }
            for record in _records.values()
            if not record.get("required")
            and record.get("status") in {"running", "finalizing"}
            and record.get("session_key") == session_key
            and (since_ts is None or (record.get("dispatched_at") or 0) >= since_ts)
        ]


def join(delegation_ids: List[str], timeout: float) -> Dict[str, List[str]]:
    """Wait on a shared deadline for legacy async completion events."""
    with _records_lock:
        events = [
            (str(delegation_id), _records[str(delegation_id)].get("done_event"))
            for delegation_id in delegation_ids
            if str(delegation_id) in _records
        ]
    deadline = time.monotonic() + max(0.0, float(timeout or 0.0))
    for _delegation_id, event in events:
        if isinstance(event, threading.Event):
            event.wait(max(0.0, deadline - time.monotonic()))
    completed = [delegation_id for delegation_id, event in events if isinstance(event, threading.Event) and event.is_set()]
    return {
        "completed": completed,
        "pending": [delegation_id for delegation_id, event in events if not isinstance(event, threading.Event) or not event.is_set()],
    }


def new_delegation_id() -> str:
    """Allocate a controller id before optional live side channels."""
    return _new_delegation_id()
