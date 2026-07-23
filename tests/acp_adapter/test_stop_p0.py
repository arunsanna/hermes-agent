"""STOP correctness (Phase 0): guaranteed is_running reset + queue drain,
and a cancel guard around the same-turn delegation join barrier.

See docs/plans/stop-p0-brief.md for the full rationale and verified anchors.
"""

from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.process_registry import process_registry


class _FakeAgent:
    def __init__(self, cancel_event_to_set=None):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.runs = []
        # Simulates a STOP arriving from the client WHILE this turn is
        # in-flight on the executor thread (the real race: cancel() sets
        # state.cancel_event concurrently, then the executor result comes
        # back and prompt() checks it before the barrier).
        self._cancel_event_to_set = cancel_event_to_set

    def run_conversation(
        self, *, user_message, conversation_history, task_id, **_kwargs
    ):
        self.runs.append(user_message)
        if self._cancel_event_to_set is not None:
            self._cancel_event_to_set.set()
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"consolidated: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class _NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None

    def replace_messages(self, *_args, **_kwargs):
        return None


class _CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))

    async def request_permission(self, *_args, **_kwargs):
        return SimpleNamespace(outcome="allow")


@pytest.fixture(autouse=True)
def _clean_completion_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_prompt_agent(monkeypatch, *, connect=True, cancel_mid_turn=False):
    manager = SessionManager(agent_factory=lambda **_kwargs: None, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    fake = _FakeAgent(cancel_event_to_set=state.cancel_event if cancel_mid_turn else None)
    state.agent = fake
    conn = _CaptureConn() if connect else None
    if conn is not None:
        acp_agent.on_connect(conn)
    monkeypatch.setattr(acp_agent, "_ensure_delegation_watcher", lambda _loop: None)
    monkeypatch.setattr(
        "tools.delegate_tool._load_config",
        lambda: {
            "acp_join_same_turn": True,
            "acp_join_max_rounds": 3,
            "acp_join_timeout_seconds": 0.05,
        },
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )
    return acp_agent, state, fake, conn


# ---------------------------------------------------------------------------
# P0.1 — is_running reset + queue drain must be guaranteed, not naked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_barrier_exception_still_resets_running_and_drains_queue(
    monkeypatch,
):
    """A crash in the post-barrier body (uncaught today) must still free the
    session and drain anything queued while the turn was running."""
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
    state.queued_prompts.append("queued after crash")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom in post-barrier code")

    monkeypatch.setattr(acp_agent.session_manager, "save_session", _boom)

    with pytest.raises(RuntimeError):
        await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="do work")],
        )

    assert state.is_running is False
    assert state.current_prompt_text == ""
    assert state.queued_prompts == []
    assert fake.runs == ["do work", "queued after crash"]


# ---------------------------------------------------------------------------
# P0.2 — a cancelled turn must skip the join barrier entirely and interrupt
# any background subagents it dispatched, instead of re-running the agent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_turn_skips_barrier_and_interrupts_subagents(monkeypatch):
    acp_agent, state, fake, _conn = _make_prompt_agent(
        monkeypatch, cancel_mid_turn=True
    )

    # A pending same-turn delegation would normally trigger join() + a
    # continuation re-run of the agent.
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [{"delegation_id": "deleg_cancelled"}],
    )

    join_calls = []
    monkeypatch.setattr(
        "tools.async_delegation.join",
        lambda delegation_ids, timeout: join_calls.append(delegation_ids) or {
            "completed": list(delegation_ids),
            "pending": [],
        },
    )

    interrupt_calls = []
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session",
        lambda **kwargs: interrupt_calls.append(kwargs) or 1,
    )

    # cancel_event.set() happens inside run_conversation, simulating STOP
    # landing on the executor thread while the turn is in flight — by the
    # time the executor result comes back, prompt() must see it before the
    # barrier runs.

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    assert response.stop_reason == "cancelled"
    # Only the original turn ran — no consolidation re-run of the agent.
    assert fake.runs == ["do work"]
    assert join_calls == []
    assert len(interrupt_calls) == 1
    assert interrupt_calls[0].get("session_key") == state.session_id
    assert state.is_running is False
