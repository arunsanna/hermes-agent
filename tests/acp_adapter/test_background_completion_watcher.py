"""Background delegation completions must re-enter their ACP session.

Before this watcher, ``delegate_task(background=true)`` results pushed onto
``process_registry.completion_queue`` were never consumed inside the ACP
process — children finishing after their turn ended were silently lost.
"""

import queue
from types import SimpleNamespace

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


class FakePR:
    def __init__(self):
        self.completion_queue = queue.Queue()


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


class NoSaveSessionManager(SessionManager):
    def __init__(self):
        super().__init__(agent_factory=lambda **_: SimpleNamespace(), db=NoopDb())
        self.saved = []

    def save_session(self, session_id):
        self.saved.append(session_id)
        return True


def make_agent_with_session(session_id="sess-1", is_running=False):
    manager = NoSaveSessionManager()
    state = SessionState(session_id=session_id, agent=SimpleNamespace())
    state.is_running = is_running
    manager._sessions[session_id] = state
    agent = HermesACPAgent(session_manager=manager)
    agent._conn = CaptureConn()
    return agent, state, manager


def completion_event(session_key="sess-1", status="completed", **extra):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_test1",
        "session_key": session_key,
        "goal": "Write the report",
        "status": status,
        "summary": "Report written: OMEGA",
        "error": None,
        "api_calls": 3,
        "duration_seconds": 12.5,
    }
    evt.update(extra)
    return evt


def formatter(evt):
    return f"[IMPORTANT: delegation {evt['delegation_id']} {evt['status']}: {evt['summary']}]"


@pytest.mark.asyncio
async def test_idle_session_receives_history_and_frames():
    agent, state, manager = make_agent_with_session()
    pr = FakePR()
    pr.completion_queue.put(completion_event())

    await agent._drain_completion_queue_once(pr, formatter)

    assert pr.completion_queue.empty()
    assert len(state.history) == 1
    assert state.history[0]["role"] == "user"
    assert "OMEGA" in state.history[0]["content"]
    assert manager.saved == ["sess-1"]

    updates = agent._conn.updates
    assert len(updates) == 2
    (sid_start, start), (sid_done, done) = updates
    assert sid_start == "sess-1" and sid_done == "sess-1"
    assert start.session_update == "tool_call"
    assert start.title.startswith("background delegation completed: Write the report")
    assert start.raw_input["tool"] == "subagent"
    assert start.raw_input["arguments"]["background"] is True
    assert done.session_update == "tool_call_update"
    assert done.tool_call_id == start.tool_call_id
    assert done.status == "completed"


@pytest.mark.asyncio
async def test_busy_session_requeues_until_idle():
    agent, state, _ = make_agent_with_session(is_running=True)
    pr = FakePR()
    pr.completion_queue.put(completion_event())

    await agent._drain_completion_queue_once(pr, formatter)
    assert state.history == []
    assert pr.completion_queue.qsize() == 1, "busy-session event must be requeued"

    state.is_running = False
    await agent._drain_completion_queue_once(pr, formatter)
    assert pr.completion_queue.empty()
    assert len(state.history) == 1


@pytest.mark.asyncio
async def test_failed_delegation_marks_frame_failed():
    agent, _, _ = make_agent_with_session()
    pr = FakePR()
    pr.completion_queue.put(completion_event(status="error", summary=None, error="boom"))

    await agent._drain_completion_queue_once(pr, formatter)

    (_, _), (_, done) = agent._conn.updates
    assert done.status == "failed"


@pytest.mark.asyncio
async def test_unknown_session_event_is_dropped_not_requeued():
    agent, _, _ = make_agent_with_session()
    pr = FakePR()
    pr.completion_queue.put(completion_event(session_key="sess-elsewhere"))

    await agent._drain_completion_queue_once(pr, formatter)
    assert pr.completion_queue.empty(), "unroutable events must not spin forever"


@pytest.mark.asyncio
async def test_foreign_event_types_are_requeued():
    agent, state, _ = make_agent_with_session()
    pr = FakePR()
    pr.completion_queue.put({"type": "watch_match", "session_id": "p1"})
    pr.completion_queue.put(completion_event())

    await agent._drain_completion_queue_once(pr, formatter)

    assert len(state.history) == 1
    assert pr.completion_queue.qsize() == 1
    assert pr.completion_queue.get_nowait()["type"] == "watch_match"
