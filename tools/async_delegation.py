#!/usr/bin/env python3
"""
Async (background) delegation registry.

Backs ``delegate_task(background=true)``: the parent agent dispatches a
subagent that runs on a module-level daemon executor and returns a handle
immediately, so the user and the model can keep working while the child runs.

When the child finishes, a completion event is pushed onto the SHARED
``process_registry.completion_queue`` with ``type="async_delegation"``. The
CLI (``cli.py`` process_loop) and gateway (``_run_process_watcher`` /
``completion_queue`` drain) already poll that queue while the agent is idle
and forge a fresh user/internal turn from each event. We deliberately reuse
that rail rather than reaching into a running agent loop:

  - completions surface as a NEW turn when the agent is idle, never spliced
    between a tool result and an assistant message. That keeps strict
    message-role alternation legal and the prompt cache intact (hard
    invariant: never mutate past context).
  - we inherit the queue's de-dup, crash-recovery checkpoint, and the
    existing CLI + gateway drain wiring for free — no new drain loops in the
    two largest files in the repo.

The completion payload carries a RICH, self-contained task-source block (the
original goal, the context the parent supplied, toolsets, model, dispatch
time, status, and the full result summary). When the result re-enters the
conversation the parent may be deep in unrelated context and won't remember
why the subagent existed; the block lets it either use the result or
re-dispatch if the world has moved on.

This module owns ONLY the async lifecycle. The actual child build + run is
delegated back to ``delegate_tool._run_single_child`` via an injected
runner, so all the credential leasing, heartbeat, timeout, and result-shaping
logic stays in one place.
"""

from __future__ import annotations

import json
import logging
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

# Back-compat alias — the daemon executor now lives in tools.daemon_pool so
# other subsystems (tool_executor, memory_manager, delegate_tool, skills_hub)
# can share it. Existing imports of ``_DaemonThreadPoolExecutor`` keep working.
_DaemonThreadPoolExecutor = DaemonThreadPoolExecutor


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# A persistent daemon executor (NOT a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat the whole point of async). Workers are daemon
# threads so a hard process exit doesn't hang on an in-flight child.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.Lock()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# A pending completion whose delivery keeps failing is retried across claim
# cycles (and across restarts via restore_undelivered_completions). Cap the
# attempts so an unroutable row converges to a terminal 'dropped' state
# instead of replaying on every restart forever.
_MAX_DELIVERY_ATTEMPTS = 8
# Staleness cap for restart replay: a pending completion older than this is
# terminally dropped instead of re-run as a fresh full-context turn (see
# restore_undelivered_completions). 48h keeps overnight/weekend results
# deliverable while stopping weeks-old sessions from replaying after upgrades.
_MAX_COMPLETION_REPLAY_AGE_S = 48 * 3600.0
_DB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Stale-delegation detection (progress-based, on by default)
# ---------------------------------------------------------------------------
# A detached runner that wedges before returning (e.g. stuck inside its first
# model API call — #60203) never reaches its ``finally`` finalizer, so no
# completion event is ever published: the delegation shows "dispatched"
# forever and the owning session looks silent until a process restart. We do
# NOT fix this with a wall-clock timeout — legitimate heavy subagent work
# (deep reviews, research fan-outs, slow reasoning models) must never be
# killed for taking long (see delegate_tool.DEFAULT_CHILD_TIMEOUT rationale).
# Instead a single monitor thread watches per-dispatch PROGRESS (api-call
# count + current tool, via an injected ``progress_fn``): a child that is
# advancing is left alone forever; a child with NO progress past the stale
# threshold is interrupted, given a grace window to unwind and deliver its
# partial results through the normal finalize path, and only force-finalized
# with a terminal ``stalled`` event if it never returns.
#
# Thresholds mirror the sync-path heartbeat staleness monitor in
# delegate_tool: idle (not inside a tool) stays tight so a wedged first API
# call is caught quickly; in-tool is much higher so legitimately slow tools
# (long terminal commands, big fetches) get time to finish.
_STALE_CHECK_INTERVAL = 30.0  # seconds between monitor sweeps
_STALE_IDLE_SECONDS = 450.0  # no progress, no current tool → stalled
_STALE_IN_TOOL_SECONDS = 1200.0  # no progress while inside a tool → stalled
_STALL_GRACE_SECONDS = 120.0  # after interrupt, time for the runner to return

_monitor_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()
# Live = not yet terminal. Upstream contributes "stalling" (the monitor has
# tripped but _finalize_stalled has not run); this fork contributes "queued"
# and "slow" from required-delegation supervision. "stalled" is deliberately
# ABSENT: _finish_finalization(..., "stalled") makes it terminal, so counting
# it live would leak capacity forever.
_LIVE_DELEGATION_STATES = frozenset(
    {"queued", "running", "slow", "stalling", "finalizing"}
)

_REQUIRED_TERMINAL_STATES = frozenset(
    {"completed", "failed", "error", "timeout", "cancelled", "interrupted"}
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (async_delegation)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS async_delegations (
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
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    for name, sql_type in (
        ("owner_pid", "INTEGER"),
        ("owner_started_at", "INTEGER"),
        ("task_json", "TEXT"),
        ("delivery_claim", "TEXT"),
        ("delivery_claimed_at", "REAL"),
        # Raw api_server session id (X-Hermes-Session-Id) of the ORIGINATING
        # request — the wake self-post target. Without persisting it,
        # completions recovered after a process restart are unroutable on
        # api_server (the in-memory record that carried it is gone).
        ("origin_session_id", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every durable dispatch, completion, and delivery-claim, deferring the close
    to the garbage collector. On a long-running gateway that exhausts
    ``RLIMIT_NOFILE`` (the cron-ledger sibling of this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _capture_routing_origin() -> Dict[str, Any]:
    """Snapshot the dispatching turn's routing origin for the completion event.

    Captured on the PARENT thread at dispatch time (the daemon worker doesn't
    carry the contextvars) and persisted with the durable record, so a
    completion replayed after a restart can reconstruct a full SessionSource
    even when the session-store origin and in-memory source cache are gone.
    scope_id matters most: on a relay-fronted deployment the connector's
    fail-closed egress guard needs the tenant discriminator (or a user
    binding) to route a scoped reply; without it, post-restart scoped
    completions bounce with "target not routed to an onboarded tenant"
    (staging 2026-08-09 defect #4). Best-effort — empty values are simply
    omitted so CLI/contextvar-unaware paths persist nothing new.
    """
    origin: Dict[str, Any] = {}
    try:
        from gateway.session_context import get_session_env

        for evt_key, env_name in (
            ("scope_id", "HERMES_SESSION_SCOPE_ID"),
            ("user_id", "HERMES_SESSION_USER_ID"),
            ("user_name", "HERMES_SESSION_USER_NAME"),
        ):
            value = get_session_env(env_name, "")
            if value:
                origin[evt_key] = value
    except Exception:  # noqa: BLE001 - routing origin is additive, never fatal
        pass
    return origin


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(__import__("os").getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in (
            "goal", "goals", "context", "toolsets", "role", "model", "is_batch",
            # Routing origin (scope_id/user_id/user_name): persisted so a
            # restart-recovered completion can reconstruct a full
            # SessionSource — see _capture_routing_origin.
            "scope_id", "user_id", "user_name",
        )
        if key in record
    }
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""),
             record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
             record["dispatched_at"], now, __import__("os").getpid(),
             owner_started_at, json.dumps(task_payload),
             record.get("origin_session_id", "")),
        )
    _prune_durable_records()


def _delete_durable_delegation(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?",
            (cutoff,),
        )
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')"
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_COMPLETED)
        if excess:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
        pending_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_DURABLE_PENDING)
        if overflow:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (overflow,),
            )


def _persist_completion(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=?""",
            (event.get("status", "completed"), event.get("completed_at", now), now,
             json.dumps(event), json.dumps(result), event["delegation_id"]),
        )


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_attempts=delivery_attempts+1, updated_at=? WHERE delegation_id=?",
            (time.time(), delegation_id),
        )


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
    now = time.time()
    recovered = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id
               FROM async_delegations WHERE state IN ('running','finalizing')"""
        ).fetchall()
        for row in rows:
            (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
             pid, started, task_json, origin_session_id) = row
            live = False
            if pid:
                live = _pid_exists(int(pid))
                if live and started is not None:
                    live = get_process_start_time(int(pid)) == int(started)
            if live:
                continue
            task = json.loads(task_json or "{}")
            event = {
                "type": "async_delegation", "delegation_id": delegation_id,
                "session_key": session_key, "origin_ui_session_id": origin_ui,
                # Restore the durable wake target so completions recovered
                # after a restart remain routable to api_server sessions.
                "origin_session_id": origin_session_id or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""),
                "goals": task.get("goals"), "context": task.get("context"),
                "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": "unknown", "summary": None,
                "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
                "dispatched_at": dispatched_at, "completed_at": now,
            }
            # Routing origin persisted at dispatch (see _capture_routing_origin):
            # restores scope_id/user_id for the reconstructed SessionSource so
            # relay egress priming works after a restart.
            for _k in ("scope_id", "user_id", "user_name"):
                if task.get(_k):
                    event[_k] = task[_k]
            result = {"status": "unknown", "summary": None, "error": event["error"]}
            conn.execute(
                """UPDATE async_delegations SET state='unknown', completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (now, now, json.dumps(event), json.dumps(result), delegation_id),
            )
            recovered += 1
    return recovered


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable pending completions as fresh turns after process start.

    Every restored event is stamped ``restored=True`` (in-memory only — the
    stamp is added after the durable payload is deserialized and is never
    persisted). Restored events originate from a *previous* process, so no
    consumer in THIS process implicitly owns them: drain paths that run
    without an ownership filter (the legacy single-session behavior) must
    leave them queued for a consumer that can positively prove ownership,
    otherwise a brand-new session adopts a dead session's delegation
    results seconds after boot (#64484).

    Staleness cap: a pending completion older than
    ``_MAX_COMPLETION_REPLAY_AGE_S`` is terminally dropped instead of
    replayed. Replaying a weeks-old completion re-runs its parent session as
    a full-context turn (a July session replayed in August burned a
    102K-token context on the staging fleet) for a result nobody is waiting
    on anymore; the payload stays queryable on the dropped row.
    """
    recover_abandoned_delegations()
    now = time.time()
    restored = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, event_json, completed_at, dispatched_at
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for delegation_id, payload, completed_at, dispatched_at in rows:
            age_basis = completed_at or dispatched_at
            if age_basis and (now - age_basis) > _MAX_COMPLETION_REPLAY_AGE_S:
                conn.execute(
                    """UPDATE async_delegations SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL,
                              updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""",
                    (now, delegation_id),
                )
                logger.warning(
                    "Async delegation %s: pending completion is %.1fh old "
                    "(cap %.1fh); terminally dropping the replay (result "
                    "remains queryable).",
                    delegation_id, (now - age_basis) / 3600.0,
                    _MAX_COMPLETION_REPLAY_AGE_S / 3600.0,
                )
                continue
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
            restored += 1
    return restored


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""",
            (now, now, delegation_id),
        )
        return cur.rowcount == 1


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, delegation_id, now - 300),
        )
        return cur.rowcount == 1


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable delegation event; non-durable events need no token."""
    if evt.get("type") != "async_delegation":
        return ""
    delegation_id = str(evt.get("delegation_id") or "")
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{__import__('os').getpid()}:{uuid.uuid4().hex}"
    return claim_id if claim_completion_delivery(delegation_id, claim_id) else None


def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Release a failed delivery claim so another consumer may retry.

    Attempts are counted at claim time, so a row that keeps being claimed and
    released has burned real delivery attempts. Once the budget is exhausted
    the row converges to a terminal ``dropped`` state instead of returning to
    ``pending`` — otherwise an undeliverable completion replays on every
    gateway restart forever (restore_undelivered_completions only restores
    pending rows).
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
        )
        if capped.rowcount == 1:
            logger.warning(
                "Async delegation %s exhausted its %d delivery attempts; "
                "marking terminally dropped (result remains queryable).",
                delegation_id, _MAX_DELIVERY_ATTEMPTS,
            )
            return True
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion that can never be delivered.

    Used when the delivery target is permanently gone — the spawning session
    ended at an explicit user boundary (/new, reset) rather than a compression
    rotation. Marking the row ``dropped`` (not ``delivered``) keeps the ack
    honest, and (not ``pending``) keeps restart recovery from replaying a
    completion that will be fail-closed dropped again every time.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge acceptance for the consumer holding this claim."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        complete_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        release_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      origin_session_id
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1],
        "dispatched_at": row[2], "completed_at": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "delivery_state": row[5], "delivery_attempts": row[6],
        "origin_session_id": row[7] or "",
    }


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) the shared daemon executor.

    We never shrink — ThreadPoolExecutor can't resize — but if the configured
    cap grows between calls we rebuild a larger pool. Existing in-flight
    futures keep running on the old pool until it's garbage collected.
    """
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegation UNITS currently running.

    A unit is one dispatch: a single subagent OR a whole fan-out batch. A batch
    counts as ONE here because it occupies one async-pool slot (the capacity
    semantics ``dispatch_async_delegation_batch`` relies on). For the count of
    actual concurrent child subagents (batch expanded), use
    ``active_task_count()``.
    """
    with _records_lock:
        return sum(
            1 for r in _records.values()
            if r.get("status") in _LIVE_DELEGATION_STATES
        )


def active_for_session(origin_ui_session_id: str) -> int:
    """Number of live async delegations owned by one UI session."""
    if not origin_ui_session_id:
        return 0
    with _records_lock:
        return sum(
            1
            for r in _records.values()
            if r.get("status") in _LIVE_DELEGATION_STATES
            and str(r.get("origin_ui_session_id") or "")
            == origin_ui_session_id
        )


def active_task_count() -> int:
    """Number of async delegation TASKS (child subagents) currently running.

    Unlike ``active_count()`` (units/slots), this expands a batch to its child
    count: a running batch of N tasks contributes N, a single subagent
    contributes 1. This is the truthful "how many subagents are actually
    working right now" figure for observability, where a 3-task batch shown as
    "1" undercounts real concurrent work. Falls back to counting a batch as 1
    if its goal list is missing.
    """
    with _records_lock:
        total = 0
        for r in _records.values():
            if r.get("status") not in _LIVE_DELEGATION_STATES:
                continue
            if r.get("is_batch"):
                goals = r.get("goals")
                total += len(goals) if isinstance(goals, (list, tuple)) and goals else 1
            else:
                total += 1
        return total


def _matches_session_selectors(
    record: Dict[str, Any],
    *,
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    return (
        (origin_ui_session_id and str(record.get("origin_ui_session_id") or "") == origin_ui_session_id)
        or (session_key and str(record.get("session_key") or "") == session_key)
        or (parent_session_id and str(record.get("parent_session_id") or "") == parent_session_id)
    )


def has_live_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    """Whether a session still owns any live async delegation.

    Live = running / stalling / finalizing — the same states the reapers'
    keepalive treats as active work.
    """
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return False
    with _records_lock:
        return any(
            r.get("status") in _LIVE_DELEGATION_STATES
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
            for r in _records.values()
        )


def _record_snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a record copy without thread-only, non-serialisable fields."""
    return {
        key: value
        for key, value in record.items()
        if key not in (
            "interrupt_fn",
            "child_interrupt_fn",
            "child_terminal_fn",
            "done_event",
        )
    }


def running_for_session(
    session_key: str, since_ts: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Snapshot running delegations for one session, optionally since a time."""
    with _records_lock:
        return [
            _record_snapshot(record)
            for record in _records.values()
            if not record.get("required")
            and record.get("status") in {"running", "finalizing"}
            and record.get("session_key") == session_key
            and (
                since_ts is None
                or (record.get("dispatched_at") or 0) >= since_ts
            )
        ]


def join(delegation_ids: List[str], timeout: float) -> Dict[str, List[str]]:
    """Wait for delegation completion events using one shared deadline."""
    with _records_lock:
        events = [
            (delegation_id, _records[delegation_id].get("done_event"))
            for delegation_id in delegation_ids
            if delegation_id in _records
        ]

    deadline = time.monotonic() + max(0.0, timeout)
    for _, done_event in events:
        if isinstance(done_event, threading.Event):
            done_event.wait(max(0.0, deadline - time.monotonic()))

    completed = [
        delegation_id
        for delegation_id, done_event in events
        if isinstance(done_event, threading.Event) and done_event.is_set()
    ]
    return {
        "completed": completed,
        "pending": [
            delegation_id
            for delegation_id, done_event in events
            if not isinstance(done_event, threading.Event) or not done_event.is_set()
        ],
    }


def new_delegation_id() -> str:
    """Allocate the stable controller id before any optional side channel."""
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _new_delegation_id() -> str:
    """Backward-compatible private alias for older callers/tests."""
    return new_delegation_id()


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") in _REQUIRED_TERMINAL_STATES
        or r.get("status") in {
            "completed", "failed", "error", "timeout",
            "cancelled", "interrupted", "unknown",
        }
        if not (r.get("required") and r.get("consumed_at") is None)
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


def _current_origin_session_id() -> str:
    """Raw session id of the ORIGINATING api_server request, or ``""``.

    The obvious source — ``HERMES_SESSION_ID`` via ``get_session_env`` — is
    NOT safe to read at dispatch time: constructing a child agent
    (``agent/agent_init.py``) calls ``set_current_session_id(child.session_id)``,
    clobbering that ContextVar *and* ``os.environ`` with the subagent's
    internal ``{timestamp}_{uuid}`` id moments before the dispatch code reads
    it, so the completion wake would self-post into the subagent's own
    (unread) session instead of the spawner's.

    The request-scoped ``HERMES_SESSION_CHAT_ID`` binding survives child
    construction: ``_bind_api_server_session`` binds ``chat_id`` to the raw
    ``X-Hermes-Session-Id``, and its only writer is ``set_session_vars`` —
    ``set_current_session_id`` never touches it. Gate on the platform: on
    push platforms ``chat_id`` is a chat, not a session, so yield ``""``
    there.
    """
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") != "api_server":
            return ""
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or ""
    except Exception:
        return ""
def _clear_required_terminal_callbacks_locked(
    record: Dict[str, Any],
) -> None:
    """Release closures that retain child agents after terminal lock win."""
    record["interrupt_fn"] = None
    record["child_interrupt_fn"] = None
    record["child_terminal_fn"] = None


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.

    Parameters
    ----------
    goal, context, toolsets, role, model
        The dispatch-time task spec, captured verbatim for the rich
        completion block.
    session_key
        The gateway session_key (from ``tools.approval.get_current_session_key``)
        captured on the parent thread BEFORE dispatch, because the daemon
        worker thread won't carry the contextvar. Used to route the
        completion back to the originating session.
    parent_session_id
        The durable ``state.db`` session id of the parent agent that spawned
        the delegation. Carried on the completion event so the gateway can
        pin routing to the spawning session instead of recovering the latest
        ``ended_at IS NULL`` row for the peer tuple (#57498).
    runner
        Zero-arg callable that builds + runs the child and returns the same
        result dict ``_run_single_child`` produces. Runs on the worker thread.
    interrupt_fn
        Optional callable to signal the child to stop (used on shutdown /
        explicit cancel).
    progress_fn
        Optional zero-arg callable returning ``(token, in_tool)`` where
        ``token`` is any comparable snapshot of the child's progress (api
        call count + current tool) and ``in_tool`` says whether the child is
        currently inside a tool call. Sampled by the stale monitor; a frozen
        token past the stale threshold marks the delegation stuck (see the
        stale-detection block at the top of this module). When omitted, the
        delegation is not monitored.
    max_async_children
        Concurrency cap. When at capacity the dispatch is REJECTED (the caller
        should fall back to sync or tell the user) rather than queued, so a
        runaway model can't pile up unbounded background work.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
    delegation_id = _new_delegation_id()
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        **_capture_routing_origin(),
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "progress_fn": progress_fn,
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
        "done_event": threading.Event(),
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        running = sum(
            1 for r in _records.values()
            if r.get("status") in ("running", "stalling")
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    _persist_dispatch(record)
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = result.get("status") or "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation %s crashed", delegation_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize(delegation_id, result, status)

    try:
        # Propagate the dispatching profile so the detached child resolves
        # get_hermes_home() under the right profile.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    # finally: a failed push must still finalize, or `join()` waiters block
    # until their deadline on a delegation that is already dead.
    try:
        _push_completion_event(event_record, result, status)
    finally:
        _finish_finalization(delegation_id, status)


def _begin_finalization(
    delegation_id: str,
) -> Optional[tuple[Dict[str, Any], Optional[Callable[[], None]]]]:
    """Atomically claim terminal delivery while keeping the record active."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in ("running", "stalling"):
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        interrupt_fn = record.get("interrupt_fn")
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["progress_fn"] = None  # stop stale-monitor sampling
        event_record = dict(record)

    return event_record, interrupt_fn


def _finish_finalization(delegation_id: str, status: str) -> None:
    with _records_lock:
        record = _records.get(delegation_id)
        done_event = record.get("done_event") if record is not None else None
        if record is not None:
            record["status"] = status
        _prune_completed_locked()
    # Wake `join()` waiters last, and outside the lock. This is the single
    # terminal chokepoint for every finalize path (normal, batch, stalled),
    # so joiners can never be left waiting on a record that already ended.
    if isinstance(done_event, threading.Event):
        done_event.set()


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.
    """
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s finished but process_registry import failed; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )
        return

    summary = result.get("summary")
    error = result.get("error")
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()

    evt = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        # session_key routes the completion back to the originating gateway
        # session; empty string => CLI (single-session) path.
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
    }
    # Routing origin captured at dispatch (see _capture_routing_origin):
    # additive, lets the gateway reconstruct a full SessionSource (incl.
    # scope_id for relay tenant egress) when its own caches are cold.
    for _k in ("scope_id", "user_id", "user_name"):
        if record.get(_k):
            evt[_k] = record[_k]
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in result:
            evt[_k] = result[_k]
    _persist_completion(evt, result)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )


def dispatch_async_delegation_batch(
    *,
    goals: List[str],
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    parent_owner_token: str = "",
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
    required: bool = False,
    parent_turn_id: str = "",
    child_ids: Optional[List[str]] = None,
    child_interrupt_fn: Optional[Callable[[str], None]] = None,
    child_terminal_fn: Optional[Callable[[str, str, str], None]] = None,
    no_progress_timeout_seconds: float = 300.0,
    start_timeout_seconds: Optional[float] = None,
    in_flight_no_progress_timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit.

    Unlike ``dispatch_async_delegation`` (which backs a single subagent),
    ``runner`` here runs the entire batch — it builds and joins on every child
    in parallel and returns the combined ``{"results": [...],
    "total_duration_seconds": N}`` dict that the synchronous path would have
    returned. We occupy ONE async slot for the whole batch (the in-batch
    parallelism is bounded separately by ``max_concurrent_children``), so a
    single ``delegate_task`` fan-out never exhausts the async pool by itself.

    When the batch finishes, a SINGLE completion event is pushed onto the
    shared ``process_registry.completion_queue`` carrying the full per-task
    ``results`` list, so the consolidated summaries re-enter the conversation
    as one message once every child is done — the chat is never blocked while
    they run.

    Returns ``{"status": "dispatched", "delegation_id": ...}`` on success or
    ``{"status": "rejected", "error": ...}`` when the async pool is at
    capacity.
    """
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    n = len(goals)
    required_child_ids = [
        str(child_id)
        for child_id in (child_ids or [])
        if str(child_id)
    ]
    # A combined goal label for status listings / the completion header.
    combined_goal = (
        goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    )
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": combined_goal,
        "goals": list(goals),
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        "parent_owner_token": str(parent_owner_token or ""),
        "status": "queued" if required else "running",
        **_capture_routing_origin(),
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "done_event": threading.Event(),
        "is_batch": True,
        "progress_fn": progress_fn,
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
        "required": bool(required),
        "parent_turn_id": str(parent_turn_id or ""),
        "child_ids": required_child_ids,
        "child_supervision": {
            child_id: {
                "child_id": child_id,
                "task_index": task_index,
                "status": "queued",
                "dispatched_at": dispatched_at,
                "started_at": None,
                "last_liveness_at": dispatched_at,
                "last_meaningful_at": dispatched_at,
                "last_activity": "queued",
                "current_tool": None,
                # Set of source ids (one per descendant agent instance —
                # see note_required_progress's `in_flight_source`) that
                # currently have a provider API call in flight. A nested
                # required delegation (max_spawn_depth >= 2) can have
                # several concurrent grandchildren all writing into this
                # SAME child slot (they share one frozen
                # _required_delegation_ancestor_binding), so "in flight" is
                # a per-source membership count, not one shared bool: a
                # sibling finishing must not clear the marker while another
                # sibling is still genuinely in flight. Effective in-flight
                # state is `bool(in_flight_sources)`; current_tool stays a
                # single shared slot (last-writer-wins) — deliberately
                # unchanged, since ANY sibling holding it non-None already
                # correctly signals "someone in this subtree is in a tool",
                # which is all the ceiling selection needs from it.
                "in_flight_sources": set(),
                "progress_generation": 0,
                "completed_at": None,
                "result": None,
            }
            for task_index, child_id in enumerate(required_child_ids)
        },
        "child_interrupt_fn": child_interrupt_fn,
        "child_terminal_fn": child_terminal_fn,
        "terminalized_child_ids": [],
        "cancel_requested": False,
        "cancelled_child_ids": [],
        "consumed_at": None,
        # True while an observe_required() call has won the exclusive
        # consuming claim and is persisting the terminal observation outside
        # _records_lock — see observe_required.
        "consuming": False,
        "result": None,
        "last_liveness_at": dispatched_at,
        "last_meaningful_at": dispatched_at,
        "last_activity": "queued",
        "current_tool": None,
        "progress_generation": 0,
        "finalization_started_at": None,
        "no_progress_timeout_seconds": max(
            0.001, float(no_progress_timeout_seconds or 300.0)
        ),
        "finalization_timeout_seconds": max(
            0.001, float(no_progress_timeout_seconds or 300.0)
        ),
        "start_timeout_seconds": max(
            0.001,
            float(
                start_timeout_seconds
                if start_timeout_seconds is not None
                else (no_progress_timeout_seconds or 300.0)
            ),
        ),
        # Wider ceiling applied while a child is silently inside a tool call
        # or has a provider API call in flight — see
        # _required_child_effective_timeout_locked. Those states produce zero
        # liveness touches by design, so judging them against the tight idle
        # ceiling kills legitimate long-running work.
        "in_flight_no_progress_timeout_seconds": max(
            0.001,
            float(
                in_flight_no_progress_timeout_seconds
                if in_flight_no_progress_timeout_seconds is not None
                else 1500.0
            ),
        ),
    }
    with _records_lock:
        running = sum(
            1
            for r in _records.values()
            if r.get("status") in _LIVE_DELEGATION_STATES
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background units."
                ),
            }
        _records[delegation_id] = record

    # Required ACP work is deliberately process-local and same-turn. It must
    # never be restored into the legacy detached-completion rail after a
    # process restart, because there is no live parent turn left to consume it.
    if not required:
        _persist_dispatch(record)
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
            # Batch status: completed unless every child errored/was interrupted.
            child_results = combined.get("results") or []
            if child_results and all(
                (r.get("status") not in ("completed", "success"))
                for r in child_results
            ):
                status = "error"
            else:
                status = "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation batch %s crashed", delegation_id)
            combined = {
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize_batch(delegation_id, combined, status)

    try:
        # Propagate the dispatching profile to the detached batch children.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    if required:
        threading.Thread(
            target=_required_watchdog,
            args=(delegation_id,),
            daemon=True,
            name=f"required-watchdog-{delegation_id}",
        ).start()
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    required_action: Optional[Dict[str, Any]] = None
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        if record.get("status") in _REQUIRED_TERMINAL_STATES:
            return
        required = bool(record.get("required"))
        if required:
            child_results = combined.get("results") or []
            child_ids = list(record.get("child_ids") or [])
            child_supervision = record.get("child_supervision") or {}
            now = time.time()
            incoming_status = str(status or "").strip().lower()
            runner_failed = incoming_status not in {
                "completed", "complete", "success", "done",
            }
            for result_index, item in enumerate(child_results):
                if not isinstance(item, dict):
                    continue
                child_id = str(item.get("child_id") or "")
                if not child_id:
                    task_index = item.get("task_index")
                    if (
                        isinstance(task_index, int)
                        and 0 <= task_index < len(child_ids)
                    ):
                        child_id = str(child_ids[task_index])
                    elif result_index < len(child_ids):
                        # Legacy runners may return one ordered result per
                        # declared child without identity metadata.
                        child_id = str(child_ids[result_index])
                child = child_supervision.get(child_id)
                if not isinstance(child, dict):
                    continue
                # Controller-owned terminal state is monotonic. A targeted
                # cancel/timeout may win while the underlying child ignores
                # interrupt and returns a late "completed" result; that late
                # future must never resurrect the child or enter the observed
                # aggregate.
                controller_owned_terminal = (
                    child.get("terminal_source") == "controller"
                    or child_id
                    in {
                        str(value)
                        for value in (
                            record.get("terminalized_child_ids") or []
                        )
                    }
                )
                if (
                    child.get("status") in _REQUIRED_TERMINAL_STATES
                    and controller_owned_terminal
                ):
                    continue
                child_status = str(
                    item.get("status") or status or "completed"
                ).strip().lower()
                if child_status in {"complete", "success", "done", ""}:
                    child_status = "completed"
                elif child_status not in _REQUIRED_TERMINAL_STATES:
                    child_status = "failed"
                child["status"] = child_status
                child["completed_at"] = now
                child["last_liveness_at"] = now
                child["last_activity"] = child_status
                child["current_tool"] = None
                child["terminal_source"] = "worker"
                stored = dict(item)
                stored["child_id"] = child_id
                stored["status"] = child_status
                child["result"] = stored

            unresolved_ids = [
                str(child_id)
                for child_id in child_ids
                for child in [
                    child_supervision.get(str(child_id))
                ]
                if isinstance(child, dict)
                and child.get("status") not in _REQUIRED_TERMINAL_STATES
            ]
            # A runner/aggregator exception, or a supposedly successful
            # aggregate that omitted a declared child, fails closed. Preserve
            # already-terminal sibling evidence, but terminalize every
            # unresolved child and interrupt the still-live batch.
            runner_failed = runner_failed or bool(unresolved_ids)
            failure_reason = str(
                combined.get("error")
                or (
                    "Required delegation batch ended before every child "
                    "produced a terminal result."
                )
            )
            if runner_failed:
                for child_id in unresolved_ids:
                    child = child_supervision.get(child_id)
                    if not isinstance(child, dict):
                        continue
                    child["status"] = "failed"
                    child["completed_at"] = now
                    child["last_liveness_at"] = now
                    child["last_activity"] = failure_reason
                    child["current_tool"] = None
                    child["terminal_source"] = "controller"
                    child["result"] = {
                        "child_id": child_id,
                        "status": "failed",
                        "summary": None,
                        "error": failure_reason,
                    }

            # Rebuild the aggregate from controller-owned child evidence.
            # This replaces stale late-future entries and guarantees one
            # canonical result per declared child in stable task order.
            canonical_results = (
                [
                    _required_child_result_locked(
                        child_supervision.get(str(child_id)),
                        child_id=str(child_id),
                    )
                    for child_id in child_ids
                    if isinstance(
                        child_supervision.get(str(child_id)), dict
                    )
                ]
                if child_ids
                else [
                    dict(item)
                    for item in child_results
                    if isinstance(item, dict)
                ]
            )
            combined = dict(combined)
            combined["results"] = canonical_results
            if runner_failed:
                combined["error"] = failure_reason
            child_states = {
                str(item.get("status") or "").strip().lower()
                for item in canonical_results
                if isinstance(item, dict)
            }
            if record.get("cancel_requested"):
                status = "cancelled"
            elif runner_failed:
                status = "failed"
            elif child_states and child_states <= {"cancelled", "interrupted"}:
                status = "cancelled"
            elif "timeout" in child_states:
                status = "timeout"
            elif child_states and not (
                child_states & {"completed", "complete", "success", "done"}
            ):
                status = "failed"
            # Required completion ownership is same-turn and process-local.
            # Terminalize it entirely under one lock hold: STOP/timeout may
            # win before us, or observe this terminal state after us, but no
            # second phase can overwrite the winner.
            batch_interrupt = record.get("interrupt_fn")
            record["status"] = status
            record["completed_at"] = time.time()
            record["last_liveness_at"] = record["completed_at"]
            record["last_activity"] = status
            record["result"] = combined
            done_event = record.get("done_event")
            terminalization = (
                _claim_required_child_terminalization_locked(
                    record, unresolved_ids
                )
                if runner_failed and unresolved_ids
                else None
            )
            record["child_terminal_fn"] = None
            # Match _begin_finalization: stop stale-monitor sampling the moment
            # the record goes terminal, or the monitor keeps calling a
            # progress_fn whose child is already gone.
            record["progress_fn"] = None
            required_action = {
                "child_terminalization": terminalization,
                "child_terminal_statuses": {
                    child_id: "failed"
                    for child_id in unresolved_ids
                },
                "reason": failure_reason,
                "interrupt": (
                    batch_interrupt
                    if runner_failed and unresolved_ids
                    else None
                ),
                "done_event": done_event,
            }
            _clear_required_terminal_callbacks_locked(record)
            _prune_completed_locked()
        else:
            record["status"] = "finalizing"
            record["completed_at"] = time.time()
            record["interrupt_fn"] = None
            record["child_interrupt_fn"] = None
            record["child_terminal_fn"] = None
            # Mirrors _begin_finalization (upstream): a finalizing record must
            # stop being sampled by the stale monitor.
            record["progress_fn"] = None
            record["result"] = combined
            event_record = dict(record)

    if required_action is not None:
        _emit_required_child_terminalization(
            required_action.get("child_terminalization"),
            status=(
                required_action.get("child_terminal_statuses")
                or "failed"
            ),
            reason=str(required_action.get("reason") or "failed"),
        )
        try:
            interrupt = required_action.get("interrupt")
            if callable(interrupt):
                interrupt()
        except Exception:
            logger.debug(
                "Failed required batch interrupt failed",
                exc_info=True,
            )
        finally:
            done_event = required_action.get("done_event")
            if isinstance(done_event, threading.Event):
                done_event.set()
        return

    try:
        _push_batch_completion_event(event_record, combined, status)
    finally:
        _finish_finalization(delegation_id, status)


def _push_batch_completion_event(
    event_record: Dict[str, Any], combined: Dict[str, Any], status: str
) -> None:
    """Push a combined async-delegation batch completion event."""
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            event_record.get("delegation_id"), exc,
        )
        return

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": event_record.get("delegation_id"),
        "session_key": event_record.get("session_key", ""),
        "origin_ui_session_id": event_record.get("origin_ui_session_id", ""),
        "origin_session_id": event_record.get("origin_session_id", ""),
        "parent_session_id": event_record.get("parent_session_id"),
        "goal": event_record.get("goal", ""),
        "goals": event_record.get("goals"),
        "context": event_record.get("context"),
        "toolsets": event_record.get("toolsets"),
        "role": event_record.get("role"),
        "model": event_record.get("model"),
        "status": status,
        "is_batch": True,
        # The full per-task results list — the formatter renders a
        # consolidated multi-task block from this.
        "results": combined.get("results") or [],
        # Per-task live transcript log paths (cache/delegation/live/...).
        # They persist after completion and double as the full-fidelity
        # operational record of each child's run.
        "live_transcripts": combined.get("live_transcripts"),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }
    # Routing origin captured at dispatch (see _capture_routing_origin).
    for _k in ("scope_id", "user_id", "user_name"):
        if event_record.get(_k):
            evt[_k] = event_record[_k]
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in combined:
            evt[_k] = combined[_k]
    _persist_completion(evt, combined)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            event_record.get("delegation_id"), exc,
        )


def _ensure_stale_monitor() -> None:
    """Start (once) the module-level stale-delegation monitor thread.

    One daemon thread serves every dispatch; it exits on its own when no
    monitorable records remain, and is restarted by the next dispatch that
    carries a ``progress_fn``.
    """
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_stale_monitor_loop,
            name="async-delegate-stale-monitor",
            daemon=True,
        )
        _monitor_thread.start()


def _stale_monitor_loop() -> None:
    """Sweep running delegations for stalled progress.

    Per sweep, for every running record with a ``progress_fn``:

    - Sample ``(token, in_tool)``. A changed token refreshes the record's
      progress timestamp — a child that keeps advancing is never touched, no
      matter how long it runs.
    - A frozen token past the idle/in-tool threshold marks the record
      ``stalling``: we call ``interrupt_fn`` so a responsive-but-slow child
      can unwind and deliver its (partial) result through the normal
      ``_finalize`` path with full fidelity.
    - A ``stalling`` record whose runner still hasn't returned after the
      grace window is force-finalized with one terminal ``stalled`` event so
      the owning session hears an outcome and the async slot frees. A late
      runner return after that is ignored by ``_begin_finalization``.
    """
    while not _monitor_stop.wait(_STALE_CHECK_INTERVAL):
        now = time.time()
        stalled: List[tuple] = []  # (delegation_id, is_batch, quiet_for, in_tool)
        expired: List[str] = []  # stalling past grace → force-finalize
        any_monitorable = False
        with _records_lock:
            for record in _records.values():
                status = record.get("status")
                if status == "stalling":
                    any_monitorable = True
                    interrupted_at = record.get("_interrupted_at") or now
                    if now - interrupted_at >= _STALL_GRACE_SECONDS:
                        expired.append(record["delegation_id"])
                    continue
                if status != "running":
                    continue
                progress_fn = record.get("progress_fn")
                if progress_fn is None:
                    continue
                any_monitorable = True
                try:
                    token, in_tool = progress_fn()
                except Exception:
                    # An unreadable child must not look permanently healthy —
                    # keep the last timestamp running instead of refreshing it.
                    token, in_tool = record.get("_progress_token"), False
                if token != record.get("_progress_token"):
                    record["_progress_token"] = token
                    record["_progress_ts"] = now
                    continue
                quiet_for = now - (record.get("_progress_ts") or now)
                limit = (
                    _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
                )
                if quiet_for >= limit:
                    record["status"] = "stalling"
                    record["_interrupted_at"] = now
                    # Structured stall context for the terminal event and
                    # status listings (#51690): how long progress was frozen,
                    # which threshold applied, and whether the child was
                    # inside a tool when it went quiet.
                    record["_stall_quiet_seconds"] = round(quiet_for, 2)
                    record["_stall_threshold_seconds"] = limit
                    record["_stall_in_tool"] = bool(in_tool)
                    stalled.append(
                        (
                            record["delegation_id"],
                            bool(record.get("is_batch")),
                            quiet_for,
                            in_tool,
                        )
                    )
        for delegation_id, _is_batch, quiet_for, in_tool in stalled:
            logger.warning(
                "Async delegation %s made no progress for %.0fs "
                "(in_tool=%s) — interrupting; grace window %.0fs",
                delegation_id, quiet_for, in_tool, _STALL_GRACE_SECONDS,
            )
            with _records_lock:
                record = _records.get(delegation_id)
                fn = record.get("interrupt_fn") if record else None
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    logger.debug(
                        "Async delegation %s stall interrupt failed: %s",
                        delegation_id, exc,
                    )
        for delegation_id in expired:
            _finalize_stalled(delegation_id)
        if not any_monitorable:
            return


def _finalize_stalled(delegation_id: str) -> None:
    """Force-finalize a stalling delegation whose runner never returned."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    completed_at = event_record.get("completed_at") or time.time()
    duration = round(
        completed_at - (event_record.get("dispatched_at") or completed_at),
        2,
    )
    quiet_seconds = event_record.get("_stall_quiet_seconds")
    threshold_seconds = event_record.get("_stall_threshold_seconds")
    stall_in_tool = event_record.get("_stall_in_tool")
    error = (
        f"Async delegation {delegation_id} stalled: the detached subagent "
        "stopped making progress (no new API calls, tool activity, or "
        "streamed tokens), did not respond to interruption, and never "
        "produced a completion event. The worker may be wedged inside a "
        "model API call — this is a known failure mode of long-lived "
        "gateway processes (#60203). Re-dispatch the task if it is still "
        "needed."
    )
    logger.error(
        "Async delegation %s force-finalized as stalled after %.0fs",
        delegation_id, duration,
    )
    # Structured stall metadata (#51690): lets parents and UIs distinguish
    # a stall-monitor kill from other failures without parsing the error
    # string, mirroring the sync path's timeout_seconds/timed_out_after_
    # seconds/timeout_phase fields.
    stall_meta = {
        "stalled_after_quiet_seconds": quiet_seconds,
        "stall_threshold_seconds": threshold_seconds,
        "stall_phase": (
            "in_tool" if stall_in_tool
            else "idle" if stall_in_tool is not None
            else None
        ),
        "stall_grace_seconds": _STALL_GRACE_SECONDS,
    }
    if event_record.get("is_batch"):
        _push_batch_completion_event(
            event_record,
            {
                "results": [],
                "error": error,
                "total_duration_seconds": duration,
                **stall_meta,
            },
            "stalled",
        )
    else:
        _push_completion_event(
            event_record,
            {
                "status": "stalled",
                "summary": None,
                "error": error,
                "api_calls": 0,
                "duration_seconds": duration,
                "exit_reason": "stalled",
                **stall_meta,
            },
            "stalled",
        )
    _finish_finalization(delegation_id, "stalled")


def _children_activity_from_token(token: Any, now: float) -> Optional[List]:
    """Parse a progress token into per-child activity dicts (best-effort).

    delegate_tool's ``_batch_progress`` emits one ``(api_call_count,
    current_tool, last_activity_ts)`` tuple per child. Foreign token shapes
    (custom dispatchers) degrade to ``None`` entries rather than raising —
    the token contract is intentionally opaque to the registry.
    """
    try:
        parts = list(token)
    except TypeError:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for part in parts:
        if isinstance(part, (list, tuple)) and len(part) >= 2:
            entry: Dict[str, Any] = {
                "api_calls": part[0],
                "current_tool": part[1],
            }
            if len(part) >= 3 and isinstance(part[2], (int, float)):
                entry["seconds_since_activity"] = round(
                    max(0.0, now - float(part[2])), 1
                )
            out.append(entry)
        else:
            out.append(None)
    return out


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


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable callables
    and private monitor bookkeeping, but exposes computed live-status
    fields for UIs (#51690):

    - ``seconds_since_progress``: how long the stale monitor has seen a
      frozen progress token (running/stalling records).
    - ``children_activity``: per-child ``{api_calls, current_tool,
      seconds_since_activity}`` sampled live from the dispatch's
      ``progress_fn``.
    - ``stalled_after_quiet_seconds`` / ``stall_threshold_seconds`` /
      ``stall_in_tool``: stall context once the monitor has tripped.
    """
    now = time.time()
    samplers: Dict[str, Callable] = {}
    with _records_lock:
        items = []
        for r in _records.values():
            # Required delegations are process-local and same-turn: they are
            # joined before the reply finalizes and must never surface on the
            # legacy detached-completion rail, which is for background work
            # that re-enters the conversation on its own.
            if r.get("required"):
                continue
            item = {
                k: v
                for k, v in r.items()
                if k not in {"interrupt_fn", "progress_fn", "done_event"}
                and not k.startswith("_")
            }
            status = r.get("status")
            if status in ("running", "stalling"):
                ts = r.get("_progress_ts")
                if ts:
                    item["seconds_since_progress"] = round(now - ts, 1)
                fn = r.get("progress_fn")
                if callable(fn):
                    samplers[r["delegation_id"]] = fn
            if status in ("stalling", "stalled"):
                for src, dst in (
                    ("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                    ("_stall_threshold_seconds", "stall_threshold_seconds"),
                    ("_stall_in_tool", "stall_in_tool"),
                ):
                    if r.get(src) is not None:
                        item[dst] = r.get(src)
            items.append(item)

    # Sample live activity OUTSIDE the lock — progress_fn reads child-agent
    # attributes and must never run under _records_lock (a slow or broken
    # sampler would block every dispatch/finalize in the process).
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


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values()
            if r.get("status") in ("running", "stalling")
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_all: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def interrupt_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
    reason: str = "session_end",
) -> int:
    """Signal running async delegations owned by ONE session to stop.

    A delegation's lifecycle is bound to the session that spawned it: when
    that session ends, its in-flight background subagents must end with it —
    a completed orphan would otherwise sit on the shared completion queue
    with no live owner, either leaking into another chat or burning tokens
    with no one listening (#55578).

    Selectors (any matching field claims the record):
    - ``origin_ui_session_id``: the live TUI tab/window that commissioned it.
    - ``session_key``: the durable routing key captured at dispatch.
    - ``parent_session_id``: the spawning agent's durable session-db id —
      the right selector for gateway chats, whose ``session_key`` (the
      platform conversation key) SURVIVES a ``/new`` reset while the
      session id rotates.

    Returns how many were interrupted.
    """
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return 0
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values()
            if r.get("status") in ("running", "stalling")
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_for_session: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info(
            "Interrupted %d async delegation(s) for ending session (%s)",
            count, reason,
        )
    return count


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
        thread = _monitor_thread
        _monitor_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _records_lock:
        for record in _records.values():
            done_event = record.get("done_event")
            if isinstance(done_event, threading.Event):
                done_event.set()
        _records.clear()
