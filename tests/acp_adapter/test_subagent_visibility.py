"""Subagent visibility over ACP: per-child frames, rawInput identity,
turn-end completion flush, and the 3.14 daemon-executor regression.

Wire contract consumed by Switchboard gateway-rs (2026-07-19): dispatch
ToolCallStart carries rawInput={tool:"delegate_task", arguments:{goal/role/
tasks[]}}; each child emits exactly one ToolCallStart on subagent.start and
one final ToolCallProgress on subagent.complete (clients close an item on ANY
update, so no intermediate frames); unpaired starts are flushed completed at
turn end.
"""

from collections import deque

import pytest

import acp_adapter.events as events
from acp_adapter.events import (
    flush_open_tool_calls,
    make_step_cb,
    make_tool_progress_cb,
)
from acp_adapter.tools import _delegate_raw_input, build_tool_start
from tools.async_delegation import _DaemonThreadPoolExecutor


@pytest.fixture
def sent(monkeypatch):
    """Capture ACP updates instead of pushing them onto an event loop."""
    captured = []

    def _capture(conn, session_id, loop, update):
        captured.append(update)

    monkeypatch.setattr(events, "_send_update", _capture)
    return captured


def make_progress_cb():
    tool_call_ids = {}
    tool_call_meta = {}
    cb = make_tool_progress_cb(None, "sess-test", None, tool_call_ids, tool_call_meta)
    return cb, tool_call_ids, tool_call_meta


# ---------------------------------------------------------------------------
# 3.1 — daemon executor must actually schedule on the running Python
# ---------------------------------------------------------------------------


def test_daemon_executor_runs_tasks_on_current_python():
    # Regression: the copied <=3.13 _adjust_thread_count referenced
    # self._initializer, which Python 3.14 removed — every submit raised
    # AttributeError and delegation fell back to synchronous batches.
    with_result = []
    executor = _DaemonThreadPoolExecutor(max_workers=2)
    try:
        futures = [executor.submit(lambda v=v: v * 2) for v in (1, 2, 3)]
        with_result = [f.result(timeout=10) for f in futures]
    finally:
        executor.shutdown(wait=True)
    assert sorted(with_result) == [2, 4, 6]

    threads = getattr(executor, "_threads", set())
    assert threads, "executor never spawned worker threads"
    assert all(t.daemon for t in threads), "delegation workers must be daemon threads"


# ---------------------------------------------------------------------------
# 3.3 — delegate_task rawInput identity
# ---------------------------------------------------------------------------


def test_delegate_raw_input_batch_drops_context_and_bounds_goals():
    raw = _delegate_raw_input(
        {
            "tasks": [
                {"goal": "g" * 900, "role": "analyst", "context": "HUGE" * 5000},
                {"goal": "short", "context": "also huge"},
            ]
        }
    )
    assert raw["tool"] == "delegate_task"
    tasks = raw["arguments"]["tasks"]
    assert len(tasks) == 2
    assert len(tasks[0]["goal"]) <= 403  # 400 + ellipsis slack
    assert tasks[0]["role"] == "analyst"
    assert "context" not in tasks[0]
    assert tasks[1] == {"goal": "short"}


def test_delegate_raw_input_single_task_shape():
    raw = _delegate_raw_input({"goal": "build the report", "role": "analyst", "background": True})
    assert raw["arguments"]["goal"] == "build the report"
    assert raw["arguments"]["role"] == "analyst"
    assert raw["arguments"]["background"] is True


def test_build_tool_start_delegate_carries_raw_input():
    update = build_tool_start("tc-1", "delegate_task", {"goal": "build the report"})
    assert update.raw_input["tool"] == "delegate_task"
    assert update.raw_input["arguments"]["goal"] == "build the report"


def test_build_tool_start_other_polished_tools_unchanged():
    update = build_tool_start("tc-2", "todo", {"todos": []})
    assert update.raw_input is None


# ---------------------------------------------------------------------------
# 3.4 — per-child subagent frames
# ---------------------------------------------------------------------------


def test_subagent_lifecycle_emits_start_and_single_final_update(sent):
    cb, _ids, _meta = make_progress_cb()

    cb("subagent.start", None, "Ship the report", None, subagent_id="sub-1", task_index=0, task_count=2, model="gpt-5.6-sol")
    cb("subagent.tool", "terminal", "running tests", None, subagent_id="sub-1")
    cb("subagent.thinking", None, "deciding structure", None, subagent_id="sub-1")
    cb("subagent.complete", None, "Report shipped", None, subagent_id="sub-1", status="completed", summary="wrote 3 sections")

    assert len(sent) == 2, f"expected start+final only, got {[type(u).__name__ for u in sent]}"
    start, done = sent
    assert start.session_update == "tool_call"
    assert start.title.startswith("subagent: Ship the report")
    assert start.raw_input["tool"] == "subagent"
    assert start.raw_input["arguments"]["subagentId"] == "sub-1"
    assert start.raw_input["arguments"]["model"] == "gpt-5.6-sol"
    assert done.session_update == "tool_call_update"
    assert done.tool_call_id == start.tool_call_id
    assert done.status == "completed"
    body = "".join(str(c) for c in (done.content or []))
    assert "Report shipped" in body
    assert "wrote 3 sections" in body


def test_subagent_error_statuses_map_to_failed(sent):
    cb, _ids, _meta = make_progress_cb()
    cb("subagent.start", None, "goal", None, subagent_id="sub-2")
    cb("subagent.complete", None, "Timed out after 300s", None, subagent_id="sub-2", status="timeout")
    assert sent[-1].status == "failed"


def test_subagent_complete_without_start_is_ignored(sent):
    cb, _ids, _meta = make_progress_cb()
    cb("subagent.complete", None, "orphan", None, subagent_id="sub-3", status="completed")
    assert sent == []


def test_subagent_updates_flag_off_suppresses_frames(sent, monkeypatch):
    monkeypatch.setenv("HERMES_ACP_SUBAGENT_UPDATES", "0")
    cb, _ids, _meta = make_progress_cb()
    cb("subagent.start", None, "goal", None, subagent_id="sub-4")
    cb("subagent.complete", None, "done", None, subagent_id="sub-4", status="completed")
    assert sent == []


def test_subagent_events_do_not_leak_into_parent_tool_queue(sent):
    cb, tool_call_ids, _meta = make_progress_cb()
    cb("subagent.start", None, "goal", None, subagent_id="sub-5")
    assert tool_call_ids == {}, "child frames must not enter the parent completion FIFO"


# ---------------------------------------------------------------------------
# 3.2 — completion resilience
# ---------------------------------------------------------------------------


def test_flush_open_tool_calls_closes_started_tools(sent):
    cb, tool_call_ids, tool_call_meta = make_progress_cb()
    cb("tool.started", "terminal", "ls", {"command": "ls"})
    cb("tool.started", "read_file", None, {"path": "/tmp/x"})
    assert len(sent) == 2  # the two starts

    flushed = flush_open_tool_calls(None, "sess-test", None, tool_call_ids, tool_call_meta)

    assert flushed == 2
    assert tool_call_ids == {}
    assert tool_call_meta == {}
    finals = sent[2:]
    assert len(finals) == 2
    assert all(u.session_update == "tool_call_update" for u in finals)
    assert all(u.status == "completed" for u in finals)


def test_flush_with_nothing_open_is_noop(sent):
    assert flush_open_tool_calls(None, "sess-test", None, {}, {}) == 0
    assert sent == []


def test_step_completion_without_start_does_not_emit_or_crash(sent):
    step = make_step_cb(None, "sess-test", None, {}, {})
    step(1, [{"name": "terminal", "result": "ok", "arguments": {}}])
    assert sent == []


def test_step_completion_pairs_with_started_tool(sent):
    cb, tool_call_ids, tool_call_meta = make_progress_cb()
    cb("tool.started", "terminal", "ls", {"command": "ls"})
    step = make_step_cb(None, "sess-test", None, tool_call_ids, tool_call_meta)
    step(2, [{"name": "terminal", "result": "file-list", "arguments": {"command": "ls"}}])
    assert sent[-1].session_update == "tool_call_update"
    assert tool_call_ids == {}
