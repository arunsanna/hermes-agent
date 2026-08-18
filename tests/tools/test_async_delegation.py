"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-released workers a beat to finalize BEFORE draining, so their
    # completion events land now instead of leaking into the next test's
    # queue (worker threads push events asynchronously; a drain that races an
    # in-flight _finalize misses it).
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _drain_for(delegation_id, timeout=5.0):
    """Drain until the event for *delegation_id* appears (discarding others).

    Completion events are pushed asynchronously by worker threads, so a
    straggler from a PREVIOUS test can land after that test's teardown drain
    and leak into the current test's queue. Matching on delegation_id makes
    the assertion immune to that cross-test leak.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def test_active_for_session_counts_every_live_delegation_state():
    with ad._records_lock:
        ad._records.update(
            {
                "running": {
                    "status": "running",
                    "origin_ui_session_id": "desktop-sid",
                },
                "stalling": {
                    "status": "stalling",
                    "origin_ui_session_id": "desktop-sid",
                },
                "finalizing": {
                    "status": "finalizing",
                    "origin_ui_session_id": "desktop-sid",
                },
                "completed": {
                    "status": "completed",
                    "origin_ui_session_id": "desktop-sid",
                },
                "other-session": {
                    "status": "running",
                    "origin_ui_session_id": "other-sid",
                },
            }
        )

    assert ad.active_for_session("desktop-sid") == 3
    assert ad.active_for_session("other-sid") == 1
    assert ad.active_for_session("") == 0


def test_dispatch_returns_immediately_without_blocking():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m"}

    t0 = time.monotonic()
    res = ad.dispatch_async_delegation(
        goal="g", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=3,
    )
    elapsed = time.monotonic() - t0

    assert res["status"] == "dispatched"
    assert res["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: dispatch returned while the runner is still
    # gated (active), so it cannot have waited on the gate. The active_count
    # check is the environment-independent proof; the generous wall-clock
    # bound is a loose sanity backstop, not the primary assertion (a loaded
    # CI runner can be slow but never anywhere near the runner's 5s gate).
    assert ad.active_count() == 1
    assert elapsed < 4.0, f"dispatch blocked {elapsed:.2f}s (gate is 5s)"
    gate.set()


def test_async_executor_workers_are_daemon_threads():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done"}

    res = ad.dispatch_async_delegation(
        goal="daemon check", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert res["status"] == "dispatched"

    deadline = time.monotonic() + 2
    worker = None
    while time.monotonic() < deadline:
        worker = next(
            (t for t in threading.enumerate() if t.name.startswith("async-delegate")),
            None,
        )
        if worker is not None:
            break
        time.sleep(0.02)
    assert worker is not None
    assert worker.daemon is True
    gate.set()
    assert _drain_one() is not None


def test_completion_event_lands_on_shared_queue_with_session_key():
    def runner():
        return {"status": "completed", "summary": "the result",
                "api_calls": 3, "duration_seconds": 2.0, "model": "test-model"}

    res = ad.dispatch_async_delegation(
        goal="compute X", context="some context", toolsets=["web", "file"],
        role="leaf", model="test-model", session_key="agent:main:cli:dm:local",
        parent_session_id="20260703_parent_sid",
        runner=runner, max_async_children=3,
    )
    assert res["status"] == "dispatched"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["summary"] == "the result"
    assert evt["session_key"] == "agent:main:cli:dm:local"
    assert evt["parent_session_id"] == "20260703_parent_sid"
    assert evt["delegation_id"] == res["delegation_id"]


def test_rich_reinjection_block_is_self_contained():
    def runner():
        return {"status": "completed", "summary": "The answer is 42.",
                "api_calls": 7, "duration_seconds": 3.5, "model": "test-model"}

    ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"], role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=3,
    )
    evt = _drain_one()
    assert evt is not None
    text = format_process_notification(evt)
    assert text is not None
    for needle in [
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ]:
        assert needle in text, f"missing {needle!r}"


def test_dispatch_rejected_at_capacity():
    ev = threading.Event()

    def blocker():
        ev.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    for i in range(2):
        r = ad.dispatch_async_delegation(
            goal=f"task{i}", context=None, toolsets=None, role="leaf",
            model="m", session_key="", runner=blocker, max_async_children=2,
        )
        assert r["status"] == "dispatched"

    r3 = ad.dispatch_async_delegation(
        goal="task3", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=2,
    )
    assert r3["status"] == "rejected"
    assert "capacity reached" in r3["error"]
    ev.set()


def test_interrupt_all_signals_running_children():
    ev = threading.Event()
    interrupted = {"count": 0}
    # No short internal timeout: the blocker holds until interrupt_fn fires.
    # The old ev.wait(timeout=5) made this test a change-detector for CI
    # worker load — on a CPU-starved runner the 5s expired before
    # interrupt_all() ran, the record finalized, and interrupt_all() found
    # nothing running (n == 0). The pytest-level timeout is the real
    # runaway guard.

    def blocker():
        ev.wait(timeout=60)
        return {"status": "interrupted", "summary": None,
                "error": "cancelled"}

    def interrupt_fn():
        interrupted["count"] += 1
        ev.set()

    r = ad.dispatch_async_delegation(
        goal="long task", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=blocker,
        interrupt_fn=interrupt_fn, max_async_children=3,
    )
    n = ad.interrupt_all(reason="test")
    assert n == 1
    assert interrupted["count"] == 1
    # child still emits a completion event after interrupt. Match on THIS
    # delegation's id — straggler 'completed' events from a previous test's
    # workers can finalize after that test's teardown drain and leak into
    # this queue (observed on loaded CI workers).
    evt = _drain_for(r["delegation_id"])
    assert evt is not None
    assert evt["status"] == "interrupted"


def _fast_stale_monitor(monkeypatch, *, idle=0.15, in_tool=0.3, grace=0.15):
    """Shrink the stale-monitor cadence so tests run in milliseconds."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", idle)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", in_tool)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", grace)


def test_stalled_runner_is_interrupted_then_finalized(monkeypatch):
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    interrupted = {"count": 0}

    def stuck_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "too late"}

    def interrupt_fn():
        interrupted["count"] += 1

    res = ad.dispatch_async_delegation(
        goal="stuck child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=stuck_runner,
        interrupt_fn=interrupt_fn, max_async_children=1,
        # Frozen progress token: the child never advances an API call.
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["type"] == "async_delegation"
        assert evt["status"] == "stalled"
        assert evt["delegation_id"] == res["delegation_id"]
        assert evt["api_calls"] == 0
        assert "stalled" in evt["error"]
        # Interrupt was requested BEFORE force-finalization (grace window).
        assert interrupted["count"] >= 1
        assert ad.active_count() == 0
    finally:
        gate.set()

    # If the ignored runner eventually returns, it must not enqueue a second
    # completion for a delegation the monitor already finalized.
    assert _drain_one(timeout=0.5) is None


def test_progressing_runner_is_never_stalled(monkeypatch):
    """A child that keeps advancing is left alone no matter how long it runs."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    ticks = {"n": 0}

    def slow_but_alive_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "done", "api_calls": 7}

    def progress_fn():
        # Token advances on every sample — simulates a child making steady
        # API-call progress.
        ticks["n"] += 1
        return (ticks["n"], None), False

    res = ad.dispatch_async_delegation(
        goal="slow child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=slow_but_alive_runner,
        max_async_children=1, progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    # Run well past the (shrunk) idle threshold — several monitor sweeps.
    time.sleep(0.6)
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"
    assert evt["summary"] == "done"


def test_stalling_runner_that_honors_interrupt_keeps_its_result(monkeypatch):
    """Interrupt-responsive children finalize through the NORMAL path.

    The monitor's interrupt gives a wedged-looking child a grace window; if
    the runner returns during it, the real result (partial work, api_calls)
    is delivered instead of a synthetic stalled event.
    """
    _fast_stale_monitor(monkeypatch, grace=5.0)
    interrupted = threading.Event()

    def runner():
        # "Wedged" until interrupted, then unwinds and reports partial work.
        interrupted.wait(timeout=10)
        return {
            "status": "interrupted",
            "summary": "partial work saved",
            "api_calls": 3,
        }

    res = ad.dispatch_async_delegation(
        goal="responsive child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner,
        interrupt_fn=interrupted.set, max_async_children=1,
        progress_fn=lambda: ((3, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "interrupted"
    assert evt["summary"] == "partial work saved"
    assert evt["api_calls"] == 3
    assert ad.active_count() == 0


def test_streaming_child_counts_as_alive(monkeypatch):
    """A child mid-stream (api_call_count frozen, last_activity_ts ticking)
    must never be stalled — streamed chunks tick _touch_activity, and the
    progress token includes that timestamp (same liveness signal as the
    compaction inactivity budget, PR #71508)."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    now = {"ts": 1000.0}

    def progress_fn():
        # api_call_count and current_tool frozen (long streaming response in
        # flight), but the activity timestamp advances with every chunk.
        now["ts"] += 1.0
        return ((1, None, now["ts"]),), False

    res = ad.dispatch_async_delegation(
        goal="streaming child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: (gate.wait(timeout=10), {"status": "completed", "summary": "streamed"})[1],
        progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    time.sleep(0.6)  # several sweeps past the shrunk idle threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_stalled_event_carries_structured_stall_metadata(monkeypatch):
    """The terminal stalled event must expose machine-readable stall context
    (#51690) — quiet duration, tripped threshold, phase, grace — mirroring
    the sync path's timeout_seconds/timed_out_after_seconds/timeout_phase."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()

    res = ad.dispatch_async_delegation(
        goal="stall metadata", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: ((0, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["status"] == "stalled"
        assert evt["stalled_after_quiet_seconds"] >= 0.3  # in-tool threshold
        assert evt["stall_threshold_seconds"] == ad._STALE_IN_TOOL_SECONDS
        assert evt["stall_phase"] == "in_tool"
        assert evt["stall_grace_seconds"] == ad._STALL_GRACE_SECONDS
    finally:
        gate.set()


def test_list_async_delegations_exposes_live_activity(monkeypatch):
    """list_async_delegations must expose per-child live activity sampled
    from progress_fn plus seconds_since_progress, for /agents UIs (#51690)."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    gate = threading.Event()
    base_ts = time.time() - 12.0

    res = ad.dispatch_async_delegation(
        goal="live listing", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: (((3, "web_search", base_ts),), True),
    )
    try:
        time.sleep(0.1)  # let the monitor stamp _progress_ts at least once
        item = next(
            d for d in ad.list_async_delegations()
            if d["delegation_id"] == res["delegation_id"]
        )
        assert item["status"] == "running"
        assert item["in_tool"] is True
        assert "seconds_since_progress" in item
        (child,) = item["children_activity"]
        assert child["api_calls"] == 3
        assert child["current_tool"] == "web_search"
        assert 10.0 <= child["seconds_since_activity"] <= 20.0
        # Callables and private bookkeeping must never leak.
        assert "progress_fn" not in item
        assert "interrupt_fn" not in item
        assert not any(k.startswith("_") for k in item)
    finally:
        gate.set()


def test_in_tool_stall_uses_higher_threshold(monkeypatch):
    """A frozen child inside a tool gets the in-tool ceiling, not the idle one."""
    _fast_stale_monitor(monkeypatch, idle=0.1, in_tool=10.0, grace=0.1)
    gate = threading.Event()

    def runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "long tool finished"}

    res = ad.dispatch_async_delegation(
        goal="long tool child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner, max_async_children=1,
        # Frozen token but in_tool=True — a legitimately slow terminal
        # command / web fetch. Must NOT be stalled at the idle threshold.
        progress_fn=lambda: ((1, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    time.sleep(0.5)  # far past idle threshold, well under in-tool threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_real_process_restart_restores_owned_completion_once(tmp_path):
    """Real-import E2E: a fresh interpreter restores a prior process's result."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = r'''
import time
from tools import async_delegation as ad
r = ad.dispatch_async_delegation(
    goal="restart", context=None, toolsets=None, role="leaf", model="m",
    session_key="owner-session", parent_session_id="durable-parent",
    runner=lambda: {"status": "completed", "summary": "after restart"},
)
deadline = time.time() + 5
while ad.active_count() and time.time() < deadline:
    time.sleep(.01)
print(r["delegation_id"])
'''
    first = subprocess.run(
        [sys.executable, "-c", producer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    delegation_id = first.stdout.strip().splitlines()[-1]

    consumer = r'''
import json
from tools.process_registry import process_registry
evt = process_registry.completion_queue.get_nowait()
print(json.dumps(evt, sort_keys=True))
'''
    second = subprocess.run(
        [sys.executable, "-c", consumer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    evt = json.loads(second.stdout.strip().splitlines()[-1])
    assert evt["delegation_id"] == delegation_id
    assert evt["session_key"] == "owner-session"
    assert evt["parent_session_id"] == "durable-parent"
    assert evt["summary"] == "after restart"

    acker = f'''
from tools import async_delegation as ad
assert ad.mark_completion_delivered({delegation_id!r})
'''
    subprocess.run(
        [sys.executable, "-c", acker], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "from tools.process_registry import process_registry; print(process_registry.completion_queue.qsize())"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=15, check=True,
    )
    assert probe.stdout.strip().splitlines()[-1] == "0"


# ---------------------------------------------------------------------------
# Integration: delegate_task(background=True) routing
# ---------------------------------------------------------------------------

def test_delegate_task_background_routes_async_and_does_not_block(monkeypatch):
    """delegate_task(background=True) returns a handle without running the
    child synchronously, and the child completes on the background thread.
    A single task is dispatched as a one-item background batch unit."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)  # a sync impl would hang delegate_task here
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    # monkeypatch (not `with`) so patches outlive delegate_task's return and
    # remain active while the background worker runs.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        goal="the real task", context="ctx",
        background=True, parent_agent=parent,
    )

    import json
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"].startswith("deleg_")
    assert "detached" in parsed["note"].lower()
    assert "later turn" in parsed["note"].lower()
    assert "ACP" in parsed["note"]
    assert "awaited and consolidated" in parsed["note"]
    assert "continue working" in parsed["note"].lower()
    assert "Do not wait or poll" not in parsed["note"]
    # Non-blocking invariant: delegate_task returned while the child is STILL
    # blocked on the closed gate, so no completion event exists yet.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1  # one background batch unit, not finished

    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Single task rides the batch path → carries a 1-item results list.
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 1
    assert evt["results"][0]["summary"] == "done: the real task"
    text = format_process_notification(evt)
    assert text is not None
    assert "the real task" in text


def test_delegate_task_background_uses_live_tui_agent_session_id(monkeypatch):
    """TUI async delegation must route to the live/compressed agent id.

    Regression: delegate_task captured the stale approval/session context key
    after compression rotated parent_agent.session_id. The resulting completion
    was orphaned and could be consumed by an unrelated desktop session poller.
    """
    import json
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.approval import reset_current_session_key, set_current_session_key

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "post-compress-tip"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *a, **k: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        },
    )

    approval_token = set_current_session_key("pre-compress-parent")
    session_tokens = set_session_vars(
        source="tui",
        session_key="pre-compress-parent",
        ui_session_id="origin-tab",
    )
    try:
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)
        assert json.loads(out)["status"] == "dispatched"
        evt = _drain_one()
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["session_key"] == "post-compress-tip"
    assert evt["origin_ui_session_id"] == "origin-tab"


def test_delegate_task_background_batch_runs_as_one_unit(monkeypatch):
    """A multi-item batch with background=True dispatches the WHOLE fan-out as
    ONE background unit (one handle, one async slot). The children run in
    parallel and join; the consolidated results come back as a single
    completion event when ALL of them finish."""
    import json
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    gate = threading.Event()

    def _blocking_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)
        return {
            "task_index": task_index, "status": "completed",
            "summary": f"done: {goal}", "api_calls": 1,
            "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }

    # Use monkeypatch (not a `with` block) so the patches stay active while the
    # background worker thread runs _execute_and_aggregate AFTER delegate_task
    # has already returned.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", _blocking_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    # Goals must clear upstream's batch quality gate (>=10 chars, added in
    # 94bc3194b "validate batch task quality before spawning children"), so
    # this test exercises dispatch rather than the rejection path.
    goal_a = "audit the auth module"
    goal_b = "benchmark the cache layer"
    goal_c = "check the retry backoff"
    out = dt.delegate_task(
        tasks=[{"goal": goal_a}, {"goal": goal_b}, {"goal": goal_c}],
        background=True,
        parent_agent=parent,
    )

    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["count"] == 3
    assert parsed["delegation_id"].startswith("deleg_")
    assert parsed["goals"] == [goal_a, goal_b, goal_c]
    # ONE background unit for the whole fan-out (not three), and the call
    # returned while all children are still blocked → chat not blocked.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1

    # Release the children; the whole batch joins and emits ONE event.
    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 3
    summaries = sorted(r["summary"] for r in evt["results"])
    assert summaries == sorted(f"done: {g}" for g in (goal_a, goal_b, goal_c))
    # The consolidated notification names all three tasks in one block.
    text = format_process_notification(evt)
    assert text is not None
    assert "TASK 1/3" in text and "TASK 2/3" in text and "TASK 3/3" in text
    assert all(f"done: {g}" in text for g in (goal_a, goal_b, goal_c))
    # No more events — it's a single combined completion, not N of them.
    assert _drain_one() is None


def test_model_dispatch_runs_required_acp_delegations_under_controller():
    """Top-level ACP delegations wait for their results before the model resumes.

    Other top-level surfaces preserve upstream detached-background behavior,
    while orchestrator children continue to wait synchronously for workers.
    """
    import tools.delegate_tool as dt
    from unittest.mock import MagicMock

    top = MagicMock()
    top._delegate_depth = 0
    top.platform = "cli"
    sub = MagicMock()
    sub._delegate_depth = 1
    sub.platform = "cli"
    acp = MagicMock()
    acp._delegate_depth = 0
    acp.platform = "acp"

    # Registry-fallback helper: ordinary top-level runs remain background.
    assert dt._model_background_value({"goal": "x"}, top) is True
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, top
    ) is True
    assert dt._model_background_value({"tasks": [{"goal": "a"}]}, top) is True

    # Workers remain synchronous. ACP roots dispatch under the required
    # same-turn controller, which gates the next provider call.
    assert dt._model_background_value({"goal": "x"}, sub) is False
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, sub
    ) is False
    assert dt._model_background_value({"goal": "x"}, acp) is True
    assert dt._model_required_value(acp) is True
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, acp
    ) is True


def test_run_agent_dispatch_marks_required_acp_delegations():
    """The live dispatch path uses the same ACP finalization contract."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0
        platform = "cli"

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        agent = _FakeAgent()
        run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "x"})
        assert captured["background"] is True

        run_agent.AIAgent._dispatch_delegate_task(
            agent, {"tasks": [{"goal": "a"}, {"goal": "b"}]}
        )
        assert captured["background"] is True

        sub = _FakeAgent()
        sub._delegate_depth = 1
        run_agent.AIAgent._dispatch_delegate_task(sub, {"goal": "x"})
        assert captured["background"] is False
        assert captured["required"] is False

        acp = _FakeAgent()
        acp.platform = "acp"
        run_agent.AIAgent._dispatch_delegate_task(acp, {"goal": "x"})
        assert captured["background"] is True
        assert captured["required"] is True

        run_agent.AIAgent._dispatch_delegate_task(
            acp,
            {"tasks": [{"goal": "a"}, {"goal": "b"}]},
        )
        assert captured["background"] is True
        assert captured["required"] is True


def test_dispatch_never_forwards_model_toolsets():
    """The model has no toolsets argument — subagents always inherit the
    parent's toolsets. Even if a model smuggles a `toolsets` key into the
    tool-call args, the live dispatch path must NOT forward it to
    delegate_task (which no longer accepts it) and must not crash."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            _FakeAgent(), {"goal": "x", "toolsets": ["web", "terminal"]}
        )
    assert "toolsets" not in captured


def test_required_rejection_never_runs_child_inline(monkeypatch):
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent.platform = "acp"
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent.session_id = "parent"
    parent._current_turn_id = "turn"
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._subagent_id = "child-1"
    child._delegate_role = "leaf"
    child_events = []
    child.tool_progress_callback = (
        lambda event_type, *_args, **_kwargs: child_events.append(event_type)
    )

    def _build(**_kwargs):
        parent._active_children.append(child)
        return child

    ran = []

    def _run(*_args, **_kwargs):
        ran.append(True)
        raise AssertionError("required rejection ran child inline")

    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(dt, "_run_single_child", _run)
    monkeypatch.setattr(
        dt, "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None, "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr(
        ad,
        "dispatch_async_delegation_batch",
        lambda **_kwargs: {"status": "rejected", "error": "capacity"},
    )

    result = json.loads(dt.delegate_task(
        goal="work", background=True, required=True, parent_agent=parent
    ))
    assert "could not be started under supervision" in result["error"]
    assert ran == []
    assert child not in parent._active_children
    child.interrupt.assert_not_called()
    child.close.assert_called_once()
    assert child_events == []


def test_required_dispatch_plumbs_configured_timeouts(monkeypatch):
    """Fix 2: delegate_task's required dispatch must read the three timeout
    knobs from delegation config (config.yaml > env > default) instead of
    letting dispatch_async_delegation_batch silently fall back to its own
    hardcoded defaults."""
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent.platform = "acp"
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent.session_id = "parent"
    parent._current_turn_id = "turn"
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._subagent_id = "child-1"
    child._delegate_role = "leaf"
    child.tool_progress_callback = lambda *_a, **_k: None

    monkeypatch.setattr(
        dt, "_build_child_agent",
        lambda **_kwargs: (parent._active_children.append(child), child)[1],
    )
    monkeypatch.setattr(
        dt, "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None, "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr(
        dt, "_load_config",
        lambda: {
            "required_no_progress_timeout_seconds": 11,
            "required_start_timeout_seconds": 22,
            "required_in_flight_timeout_seconds": 33,
        },
    )
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return {"status": "rejected", "error": "capture-only"}

    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", _capture)

    dt.delegate_task(goal="work", background=True, required=True, parent_agent=parent)

    assert captured["no_progress_timeout_seconds"] == 11.0
    assert captured["start_timeout_seconds"] == 22.0
    assert captured["in_flight_no_progress_timeout_seconds"] == 33.0


def test_required_dispatch_timeout_defaults_when_unconfigured(monkeypatch):
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent.platform = "acp"
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent.session_id = "parent"
    parent._current_turn_id = "turn"
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._subagent_id = "child-1"
    child._delegate_role = "leaf"
    child.tool_progress_callback = lambda *_a, **_k: None

    monkeypatch.setattr(
        dt, "_build_child_agent",
        lambda **_kwargs: (parent._active_children.append(child), child)[1],
    )
    monkeypatch.setattr(
        dt, "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None, "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.delenv("DELEGATION_REQUIRED_NO_PROGRESS_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DELEGATION_REQUIRED_START_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DELEGATION_REQUIRED_IN_FLIGHT_TIMEOUT_SECONDS", raising=False)
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return {"status": "rejected", "error": "capture-only"}

    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", _capture)

    dt.delegate_task(goal="work", background=True, required=True, parent_agent=parent)

    assert captured["no_progress_timeout_seconds"] == 300.0
    assert captured["start_timeout_seconds"] == 300.0
    assert captured["in_flight_no_progress_timeout_seconds"] == 1500.0


def test_required_inner_submit_failure_disposes_every_unstarted_child(
    monkeypatch,
):
    import tools.daemon_pool as daemon_pool
    import tools.delegate_tool as dt
    import tools.delegation_live_log as live_log

    class _FailingInnerExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            raise RuntimeError("inner submit failed")

        def shutdown(self, *args, **kwargs):
            return None

    class _Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        _parent_subagent_id = None
        tool_progress_callback = None

        def __init__(self, child_id):
            self._subagent_id = child_id
            self.close_count = 0
            self.run_count = 0

        def close(self):
            self.close_count += 1

        def run_conversation(self, **_kwargs):
            self.run_count += 1
            raise AssertionError("unstarted child reached provider")

    parent = SimpleNamespace(
        platform="acp",
        _delegate_depth=0,
        _interrupt_requested=False,
        session_id="parent-submit-failure",
        _current_turn_id="turn-submit-failure",
        _required_delegation_owner_token="owner-submit-failure",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=None,
        _delegate_spinner=None,
        _memory_manager=None,
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    children = [_Child("child-a"), _Child("child-b")]

    def _build(**kwargs):
        child = children[len(parent._active_children)]
        parent._active_children.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(
        daemon_pool, "DaemonThreadPoolExecutor", _FailingInnerExecutor
    )
    monkeypatch.setattr(
        live_log,
        "create_live_transcripts",
        lambda *_args, **_kwargs: (None, [], []),
    )
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    # Goals must clear upstream's >=10-char batch quality gate (94bc3194b) so
    # this exercises dispatch rather than the rejection path.
    dispatched = json.loads(dt.delegate_task(
        tasks=[{"goal": "audit module a"}, {"goal": "audit module b"}],
        background=True,
        required=True,
        parent_agent=parent,
    ))
    terminal = ad.wait_required(
        parent, dispatched["delegation_id"], timeout_seconds=2
    )

    assert terminal["status"] == "failed"
    assert {
        item["child_id"] for item in terminal["result"]["results"]
    } == {"child-a", "child-b"}
    assert [child.run_count for child in children] == [0, 0]
    assert [child.close_count for child in children] == [1, 1]
    assert parent._active_children == []


def test_terminal_required_child_never_starts_provider():
    import tools.delegate_tool as dt

    owner = SimpleNamespace(
        platform="acp",
        session_id="parent-late-start",
        _current_turn_id="turn-late-start",
        _required_delegation_owner_token="owner-late-start",
    )
    release = threading.Event()
    def _blocked_runner():
        release.wait(timeout=5)
        return {"results": [], "total_duration_seconds": 0}

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["late child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=owner.session_id,
        parent_session_id=owner.session_id,
        parent_owner_token=owner._required_delegation_owner_token,
        parent_turn_id=owner._current_turn_id,
        runner=_blocked_runner,
        required=True,
        child_ids=["child-late"],
        max_async_children=3,
    )
    delegation_id = dispatch["delegation_id"]
    ad.cancel_required(owner, delegation_id)

    provider_calls = []
    child_events = []
    close_calls = []
    child = SimpleNamespace(
        _required_delegation_id=delegation_id,
        _subagent_id="child-late",
        _delegate_role="leaf",
        _delegate_saved_tool_names=[],
        tool_progress_callback=(
            lambda event_type, *_args, **_kwargs: child_events.append(
                event_type
            )
        ),
        run_conversation=lambda **_kwargs: provider_calls.append(True),
        close=lambda: close_calls.append(True),
    )
    owner._active_children = [child]
    owner._active_children_lock = threading.Lock()
    try:
        result = dt._run_single_child(
            0, "late child", child=child, parent_agent=owner
        )
    finally:
        release.set()

    assert result["status"] == "cancelled"
    assert result["api_calls"] == 0
    assert provider_calls == []
    assert "subagent.start" not in child_events
    assert close_calls == [True]
    assert owner._active_children == []


def test_target_cancelled_child_never_starts_while_sibling_continues():
    import tools.delegate_tool as dt

    owner = SimpleNamespace(
        platform="acp",
        session_id="parent-target-late-start",
        _current_turn_id="turn-target-late-start",
        _required_delegation_owner_token="owner-target-late-start",
    )
    release = threading.Event()

    def _blocked_runner():
        release.wait(timeout=5)
        return {"results": [], "total_duration_seconds": 0}

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["active sibling", "queued child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=owner.session_id,
        parent_session_id=owner.session_id,
        parent_owner_token=owner._required_delegation_owner_token,
        parent_turn_id=owner._current_turn_id,
        runner=_blocked_runner,
        required=True,
        child_ids=["child-active", "child-target"],
        max_async_children=3,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="child-active",
        current_tool=None,
        activity="active sibling started",
        meaningful=False,
        state="running",
    )
    cancelled = ad.cancel_required(
        owner, delegation_id, child_id="child-target"
    )
    assert cancelled["terminal"] is False
    assert cancelled["status"] == "running"

    provider_calls = []
    child_events = []
    child = SimpleNamespace(
        _required_delegation_id=delegation_id,
        _subagent_id="child-target",
        _delegate_role="leaf",
        _delegate_saved_tool_names=[],
        tool_progress_callback=(
            lambda event_type, *_args, **_kwargs: child_events.append(
                event_type
            )
        ),
        run_conversation=lambda **_kwargs: provider_calls.append(True),
    )
    try:
        result = dt._run_single_child(
            1, "queued child", child=child, parent_agent=owner
        )
    finally:
        ad.cancel_required(owner, delegation_id)
        release.set()

    assert result["status"] == "cancelled"
    assert result["api_calls"] == 0
    assert provider_calls == []
    assert "subagent.start" not in child_events


def test_required_whole_cancel_releases_worker_while_child_is_uninterruptible(
    monkeypatch,
):
    import tools.delegate_tool as dt
    import tools.delegation_live_log as live_log

    stuck_started = threading.Event()
    stuck_release = threading.Event()
    next_started = threading.Event()
    build_count = 0

    class _Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        _parent_subagent_id = None

        def __init__(self, name):
            self.name = name
            self._subagent_id = f"child-{name}"
            self.tool_progress_callback = None

        def interrupt(self, *_args):
            return None  # deliberately ignore cancellation

        def close(self):
            return None

    parent = SimpleNamespace(
        platform="acp",
        _delegate_depth=0,
        _interrupt_requested=False,
        session_id="parent-cancel-capacity",
        _current_turn_id="turn-cancel-capacity",
        _required_delegation_owner_token="owner-cancel-capacity",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=None,
        _delegate_spinner=None,
        _memory_manager=None,
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )

    def _build(**_kwargs):
        nonlocal build_count
        build_count += 1
        child = _Child("stuck" if build_count == 1 else "next")
        parent._active_children.append(child)
        return child

    def _run(task_index, goal, child=None, **_kwargs):
        if child.name == "stuck":
            stuck_started.set()
            stuck_release.wait(timeout=20)
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": "late",
            }
        next_started.set()
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "next ran",
        }

    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(dt, "_run_single_child", _run)
    monkeypatch.setattr(dt, "_apply_summary_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(dt, "_get_max_async_children", lambda: 1)
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 1)
    monkeypatch.setattr(live_log, "create_live_transcripts", lambda *_a, **_k: (None, [], []))
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    try:
        first = json.loads(
            dt.delegate_task(
                goal="stuck",
                background=True,
                required=True,
                parent_agent=parent,
            )
        )
        assert stuck_started.wait(timeout=2)
        cancelled = ad.cancel_required(parent, first["delegation_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["terminal"] is True

        second = json.loads(
            dt.delegate_task(
                goal="next",
                background=True,
                required=True,
                parent_agent=parent,
            )
        )
        assert second["status"] == "dispatched"
        assert next_started.wait(timeout=3), (
            "cancelled required batch retained the only async worker"
        )
    finally:
        stuck_release.set()


def test_required_target_cancel_abandons_only_stuck_child_and_keeps_sibling(
    monkeypatch,
):
    import tools.delegate_tool as dt
    import tools.delegation_live_log as live_log

    stuck_started = threading.Event()
    stuck_release = threading.Event()
    sibling_done = threading.Event()
    built = []

    class _Child:
        _delegate_role = "leaf"
        _delegate_depth = 1
        _parent_subagent_id = None

        def __init__(self, index):
            self.index = index
            self._subagent_id = f"child-{index}"
            self.tool_progress_callback = None

        def interrupt(self, *_args):
            return None

        def close(self):
            return None

    parent = SimpleNamespace(
        platform="acp",
        _delegate_depth=0,
        _interrupt_requested=False,
        session_id="parent-target-cancel",
        _current_turn_id="turn-target-cancel",
        _required_delegation_owner_token="owner-target-cancel",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=None,
        _delegate_spinner=None,
        _memory_manager=None,
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )

    def _build(task_index, **_kwargs):
        child = _Child(task_index)
        built.append(child)
        parent._active_children.append(child)
        return child

    def _run(task_index, goal, child=None, **_kwargs):
        if child.index == 1:
            stuck_started.set()
            stuck_release.wait(timeout=20)
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": "late stuck result",
            }
        sibling_done.set()
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "useful sibling",
        }

    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(dt, "_run_single_child", _run)
    monkeypatch.setattr(dt, "_apply_summary_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(dt, "_get_max_async_children", lambda: 1)
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr(live_log, "create_live_transcripts", lambda *_a, **_k: (None, [], []))
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    try:
        dispatched = json.loads(
            dt.delegate_task(
                # >=10 chars to clear upstream's batch quality gate (94bc3194b).
                tasks=[{"goal": "fast sibling task"}, {"goal": "stuck child task"}],
                background=True,
                required=True,
                parent_agent=parent,
            )
        )
        assert stuck_started.wait(timeout=2)
        assert sibling_done.wait(timeout=2)
        cancelled = ad.cancel_required(
            parent,
            dispatched["delegation_id"],
            child_id="child-1",
        )
        assert {
            item["child_id"]: item["status"]
            for item in cancelled["children"]
        }["child-1"] == "cancelled"

        terminal = ad.wait_required(
            parent, dispatched["delegation_id"], timeout_seconds=3
        )
        assert terminal["terminal"] is True
        result_by_index = {
            item["task_index"]: item for item in terminal["result"]["results"]
        }
        assert result_by_index[0]["summary"] == "useful sibling"
        assert result_by_index[1]["status"] == "cancelled"
    finally:
        stuck_release.set()


def test_required_stop_during_registration_cancels_without_running_child(monkeypatch):
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent.platform = "acp"
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent.session_id = "parent"
    parent._current_turn_id = "turn"
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._subagent_id = "child-1"
    child._delegate_role = "leaf"

    def _build(**_kwargs):
        parent._active_children.append(child)
        return child

    ran = []

    def _register(**_kwargs):
        assert child in parent._active_children
        parent._interrupt_requested = True
        return {"status": "rejected", "error": "stopped during registration"}

    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(
        dt, "_run_single_child", lambda *_a, **_k: ran.append(True)
    )
    monkeypatch.setattr(
        dt, "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": None, "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", _register)

    result = json.loads(dt.delegate_task(
        goal="work", background=True, required=True, parent_agent=parent
    ))
    assert ran == []
    assert "could not be started under supervision" in result["error"]
    assert child not in parent._active_children
    child.interrupt.assert_not_called()
    child.close.assert_called_once()


def test_required_controller_identity_survives_live_log_creation_failure(
    monkeypatch,
):
    import tools.delegate_tool as dt
    import tools.delegation_live_log as live_log

    release = threading.Event()
    heartbeat_seen = threading.Event()
    heartbeat = {}
    heartbeat_count = []
    parent_touches = []
    parent_touch_seen = threading.Event()
    interrupts = []

    class _Child:
        _subagent_id = "child-live-log-failure"
        _delegate_role = "leaf"
        _delegate_depth = 1
        _parent_subagent_id = None
        _credential_pool = None
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_reasoning_tokens = 0
        session_estimated_cost_usd = 0.0
        model = "fake-child"

        def __init__(self):
            self._activity_tick = 0
            self.tool_progress_callback = self._progress

        def _progress(self, event_type, *args, **kwargs):
            if event_type == "subagent.heartbeat":
                heartbeat.update(kwargs)
                heartbeat_count.append(1)
                heartbeat_seen.set()

        def get_activity_summary(self):
            self._activity_tick += 1
            return {
                "current_tool": "read_file",
                "api_call_count": 1,
                "max_iterations": 10,
                "last_activity_ts": float(self._activity_tick),
                "last_activity_desc": "reading",
            }

        def run_conversation(self, **_kwargs):
            release.wait(timeout=2)
            return {
                "final_response": "done",
                "messages": [],
                "completed": True,
                "api_calls": 1,
            }

        def interrupt(self, *_args):
            # Deliberately ignore controller interruption. The heartbeat
            # must still stop independently while this worker remains live.
            interrupts.append("ignored")

        def close(self):
            return None

    child = _Child()
    def _touch_parent(_description, **_kwargs):
        parent_touches.append(1)
        parent_touch_seen.set()

    parent = SimpleNamespace(
        platform="acp",
        _delegate_depth=0,
        _interrupt_requested=False,
        session_id="parent-live-log-failure",
        _current_turn_id="turn-live-log-failure",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=None,
        _delegate_spinner=None,
        _memory_manager=None,
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
        _touch_activity=_touch_parent,
    )

    def _build(**_kwargs):
        parent._active_children.append(child)
        return child

    monkeypatch.setattr(
        live_log,
        "create_live_transcripts",
        lambda *_args, **_kwargs: (None, [], []),
    )
    monkeypatch.setattr(dt, "_build_child_agent", _build)
    monkeypatch.setattr(dt, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    try:
        dispatched = json.loads(dt.delegate_task(
            goal="work",
            background=True,
            required=True,
            parent_agent=parent,
        ))
        assert dispatched["status"] == "dispatched"
        assert heartbeat_seen.wait(timeout=1)
        assert parent_touch_seen.wait(timeout=1)
        controller_id = dispatched["delegation_id"]
        controller = ad.required_status(parent, controller_id)
        assert child._required_delegation_id == controller_id
        assert heartbeat["supervision_status"] != "detached"
        assert controller["status"] == "running"
        assert controller["progress_generation"] >= 1

        with ad._records_lock:
            record = ad._records[controller_id]
            record["no_progress_timeout_seconds"] = 0.01
            # _Child.get_activity_summary() always reports current_tool=
            # "read_file", so the heartbeat puts this child under the wider
            # in-flight ceiling (state-aware timeouts) — tighten it too so
            # the deadline this test forces still fires.
            record["in_flight_no_progress_timeout_seconds"] = 0.01
            record["child_supervision"][child._subagent_id][
                "last_meaningful_at"
            ] -= 1
            done_event = record["done_event"]
        heartbeats_before_timeout = len(heartbeat_count)
        touches_before_timeout = len(parent_touches)
        assert done_event.wait(timeout=1)
        assert ad.required_status(parent, controller_id)["status"] == "timeout"
        threading.Event().wait(0.06)
        assert len(heartbeat_count) == heartbeats_before_timeout
        assert len(parent_touches) == touches_before_timeout
        assert interrupts == ["ignored"]
    finally:
        release.set()


def test_delegate_task_background_detaches_child_from_parent(monkeypatch):
    """A background child must NOT remain in parent._active_children —
    otherwise parent-turn interrupts / cache evicts / session close would
    kill the detached subagent mid-run."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)
        return {"task_index": 0, "status": "completed", "summary": "ok"}

    def build_and_register(**kw):
        # Mirror what the real _build_child_agent does: register the child
        # for interrupt propagation.
        parent._active_children.append(fake_child)
        return fake_child

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    with patch.object(dt, "_build_child_agent", side_effect=build_and_register), \
         patch.object(dt, "_run_single_child", side_effect=slow_child), \
         patch.object(dt, "_resolve_delegation_credentials", return_value=creds):
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)

    import json
    assert json.loads(out)["status"] == "dispatched"
    # Child detached immediately at dispatch, while it is still running.
    assert fake_child not in parent._active_children
    gate.set()
    assert _drain_one() is not None


def test_concurrent_dispatch_respects_capacity():
    """Two threads racing dispatch with cap=1 must yield exactly one accept
    (capacity check and record insert are atomic under the records lock)."""
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    results = []
    barrier = threading.Barrier(2)

    def racer():
        barrier.wait(timeout=5)
        results.append(
            ad.dispatch_async_delegation(
                goal="race", context=None, toolsets=None, role="leaf",
                model="m", session_key="", runner=blocker,
                max_async_children=1,
            )
        )

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["dispatched", "rejected"]
    gate.set()


# ---------------------------------------------------------------------------
# Gateway routing: session_key -> platform/chat_id, rich formatting, injection
# ---------------------------------------------------------------------------

def _make_async_evt(**over):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x1",
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "context": "repo /tmp/p",
        "toolsets": ["terminal"],
        "role": "leaf",
        "model": "m",
        "status": "completed",
        "summary": "Found the bug in test_foo",
        "api_calls": 4,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_gateway_formatter_renders_async_block():
    from gateway.run import _format_gateway_process_notification

    txt = _format_gateway_process_notification(_make_async_evt())
    assert txt is not None
    assert "ASYNC DELEGATION COMPLETE" in txt
    assert "Found the bug in test_foo" in txt
    assert "Investigate flaky test" in txt


def test_gateway_cli_origin_event_left_unrouted():
    """An empty session_key (CLI origin) is left without routing fields."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt(session_key="")
    runner._enrich_async_delegation_routing(evt)
    assert "platform" not in evt



def test_running_for_session_filters_by_session_and_since_timestamp():
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    old = ad.dispatch_async_delegation(
        goal="old", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a", runner=blocker, max_async_children=3,
    )
    since = time.time()
    other = ad.dispatch_async_delegation(
        goal="other", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-b", runner=blocker, max_async_children=3,
    )
    current = ad.dispatch_async_delegation(
        goal="current", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a", runner=blocker, max_async_children=3,
    )

    all_for_session = ad.running_for_session("session-a")
    current_for_session = ad.running_for_session("session-a", since)

    assert {record["delegation_id"] for record in all_for_session} == {
        old["delegation_id"], current["delegation_id"],
    }
    assert [record["delegation_id"] for record in current_for_session] == [
        current["delegation_id"]
    ]
    assert other["delegation_id"] not in {
        record["delegation_id"] for record in all_for_session
    }
    assert all("interrupt_fn" not in record for record in all_for_session)
    assert all("done_event" not in record for record in all_for_session)
    gate.set()


def test_join_reports_completed_and_pending_with_one_shared_deadline():
    completed_gate = threading.Event()
    pending_gate = threading.Event()

    def complete_runner():
        completed_gate.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    def pending_runner():
        pending_gate.wait(timeout=5)
        return {"status": "completed", "summary": "late"}

    completed = ad.dispatch_async_delegation(
        goal="completed", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a", runner=complete_runner,
        max_async_children=2,
    )
    pending = ad.dispatch_async_delegation(
        goal="pending", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a", runner=pending_runner,
        max_async_children=2,
    )
    completed_gate.set()

    started = time.monotonic()
    joined = ad.join(
        [completed["delegation_id"], pending["delegation_id"]], timeout=0.05
    )
    elapsed = time.monotonic() - started

    assert joined == {
        "completed": [completed["delegation_id"]],
        "pending": [pending["delegation_id"]],
    }
    assert elapsed < 0.2
    pending_gate.set()


def test_done_event_is_set_only_after_completion_event_is_enqueued(monkeypatch):
    observed_during_put = []
    real_queue = process_registry.completion_queue

    class _ObservingQueue:
        def put(self, event):
            with ad._records_lock:
                done_event = ad._records[event["delegation_id"]]["done_event"]
                observed_during_put.append(done_event.is_set())
            real_queue.put(event)

        def empty(self):
            return real_queue.empty()

        def get_nowait(self):
            return real_queue.get_nowait()

    monkeypatch.setattr(process_registry, "completion_queue", _ObservingQueue())
    result = ad.dispatch_async_delegation(
        goal="ordered", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a",
        runner=lambda: {"status": "completed", "summary": "done"},
        max_async_children=1,
    )

    assert _drain_one() is not None
    with ad._records_lock:
        done_event = ad._records[result["delegation_id"]]["done_event"]
    assert observed_during_put == [False]
    assert done_event.is_set()
    assert "done_event" not in ad.list_async_delegations()[0]


def test_single_task_truncation_banner_when_max_iterations():
    """A single async subagent that hit its iteration cap (exit_reason=
    max_iterations) must surface a TRUNCATED marker in the formatted result,
    even though status stays 'completed' (a summary exists)."""
    evt = _make_async_evt(
        status="completed",
        summary="Did part of the work then ran out of budget.",
        exit_reason="max_iterations",
    )
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" in text
    assert "max_iterations" in text
    # The summary is still shown, just flagged.
    assert "Did part of the work" in text


def test_single_task_no_banner_when_clean():
    """A cleanly-finished subagent must NOT get a truncation banner."""
    evt = _make_async_evt(status="completed", summary="All done.", exit_reason="completed")
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" not in text


def test_batch_truncation_banner_marks_only_truncated_task():
    """In a batch, only the task that hit max_iterations gets the TRUNCATED
    marker; a clean sibling keeps the normal check icon."""
    evt = _make_async_evt(
        is_batch=True,
        goals=["clean task", "truncated task"],
        results=[
            {
                "task_index": 0,
                "status": "completed",
                "summary": "finished cleanly",
                "api_calls": 5,
                "exit_reason": "completed",
                "truncated": False,
            },
            {
                "task_index": 1,
                "status": "completed",
                "summary": "cut off mid-work",
                "api_calls": 250,
                "exit_reason": "max_iterations",
                "truncated": True,
            },
        ],
    )
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" in text
    # The clean task's summary and the truncated one's both render...
    assert "finished cleanly" in text
    assert "cut off mid-work" in text
    # ...but the banner is tied to the truncated task, not the clean one.
    trunc_pos = text.index("cut off mid-work")
    clean_pos = text.index("finished cleanly")
    banner_pos = text.index("TRUNCATED")
    # The header banner for task 2 appears after task 1's summary.
    assert banner_pos > clean_pos

