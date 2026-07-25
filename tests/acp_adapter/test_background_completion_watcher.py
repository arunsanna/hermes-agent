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


# ---------------------------------------------------------------------------
# Regression: cross-session delegation leak (#delegation-cross-session-leak,
# 2026-07-25 Switchboard incident). A delegation dispatched under session A
# must never be delivered into session B's context, even when both sessions'
# processes share one SessionDB (e.g. a host that fails to isolate
# HERMES_HOME per process) and B's process has never created or loaded A.
# ---------------------------------------------------------------------------


class ForeignSessionDb(NoopDb):
    """Simulates a SessionDB shared with ANOTHER process: it knows about a
    session (``other-session``) this process's SessionManager never created
    or loaded — exactly what ``get_session``'s DB-restore fallback would
    silently adopt."""

    def get_session(self, session_id, *_args, **_kwargs):
        if session_id == "other-session":
            return {
                "id": "other-session",
                "source": "acp",
                "model_config": "{}",
                "model": "",
                "billing_provider": None,
                "billing_base_url": None,
            }
        return None

    def get_messages_as_conversation(self, *_args, **_kwargs):
        return []


class ForeignRestoreSessionManager(NoSaveSessionManager):
    """A SessionManager whose DB WOULD restore a foreign session on
    ``get_session`` — proving ``peek_session`` (used by the watcher) refuses
    to adopt it even though the legacy code path could."""

    def __init__(self):
        SessionManager.__init__(
            self, agent_factory=lambda **_: SimpleNamespace(), db=ForeignSessionDb()
        )
        self.saved = []

    def _get_db(self):
        return self._db_instance


@pytest.mark.asyncio
async def test_cross_session_leak_is_blocked_even_when_db_would_restore_it():
    """The exact production shape: a completion addressed to a session this
    process never created/loaded must be dropped, not adopted via DB
    restore — proving the watcher can never splice one session's delegation
    result into an unrelated session's history/connection.
    """
    # Sanity check on a THROWAWAY instance: the legacy accessor WOULD have
    # adopted (and cached) this foreign session — this is what makes the
    # leak possible without the fix. Using a separate instance so this
    # assertion doesn't contaminate the manager under test below (get_session
    # caches whatever it restores into ``_sessions``, which would make a
    # subsequent ``peek_session`` on the SAME instance pass for the wrong
    # reason).
    assert ForeignRestoreSessionManager().get_session("other-session") is not None

    manager = ForeignRestoreSessionManager()
    # The strict accessor the watcher now uses must refuse the same session.
    assert manager.peek_session("other-session") is None

    agent = HermesACPAgent(session_manager=manager)
    agent._conn = CaptureConn()
    pr = FakePR()
    pr.completion_queue.put(completion_event(session_key="other-session"))

    await agent._drain_completion_queue_once(pr, formatter)

    assert pr.completion_queue.empty(), "unowned event must be dropped, not spun forever"
    assert manager.saved == [], "no foreign session state may be persisted"
    assert agent._conn.updates == [], "no notification may be sent for a session we don't own"


@pytest.mark.asyncio
async def test_durable_completion_is_not_redelivered_after_first_claim():
    """A completion already claimed/delivered by one consumer (simulating a
    second hermes-acp process racing on the same shared SessionDB) must not
    be re-spliced into history a second time.
    """
    from tools.async_delegation import _persist_dispatch, claim_completion_delivery

    agent, state, manager = make_agent_with_session(session_id="sess-1")
    record = {
        "delegation_id": "deleg_dup1",
        "session_key": "sess-1",
        "dispatched_at": 0.0,
    }
    _persist_dispatch(record)
    # Simulate another consumer having already claimed this delivery (e.g.
    # a different process's watcher tick that fired first).
    assert claim_completion_delivery("deleg_dup1", "other-consumer-claim") is True

    pr = FakePR()
    pr.completion_queue.put(completion_event(session_key="sess-1", delegation_id="deleg_dup1"))

    await agent._drain_completion_queue_once(pr, formatter)

    assert state.history == [], "already-claimed completion must not be delivered again"
    assert pr.completion_queue.empty(), "an already-claimed event should be dropped, not re-queued"
