"""ACP prompt turns join same-turn background delegations before finalizing."""

import time

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.process_registry import process_registry


class _FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.runs = []

    def run_conversation(
        self, *, user_message, conversation_history, task_id, **_kwargs
    ):
        self.runs.append(user_message)
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


@pytest.fixture(autouse=True)
def _clean_completion_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_prompt_agent(monkeypatch):
    fake = _FakeAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    monkeypatch.setattr(acp_agent, "_ensure_delegation_watcher", lambda _loop: None)
    monkeypatch.setattr(
        "tools.delegate_tool._load_config",
        lambda: {
            "acp_join_same_turn": True,
            "acp_join_max_rounds": 3,
            "acp_join_timeout_seconds": 0.05,
        },
    )
    return acp_agent, state, fake


def _completion_event(session_id):
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_same_turn",
        "session_key": session_id,
        "goal": "review the change",
        "status": "completed",
        "summary": "Reviewer found OMEGA",
        "error": None,
        "api_calls": 2,
        "duration_seconds": 0.1,
    }


@pytest.mark.asyncio
async def test_prompt_reruns_agent_to_consolidate_same_turn_delegation(
    monkeypatch,
):
    acp_agent, state, fake = _make_prompt_agent(monkeypatch)
    process_registry.completion_queue.put(_completion_event(state.session_id))
    scans = iter(
        [
            [{"delegation_id": "deleg_same_turn"}],
            [],
        ]
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: next(scans),
    )
    monkeypatch.setattr(
        "tools.async_delegation.join",
        lambda delegation_ids, timeout: {
            "completed": list(delegation_ids),
            "pending": [],
        },
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do the work")],
    )

    assert response.stop_reason == "end_turn"
    assert len(fake.runs) == 2
    assert fake.runs[0] == "do the work"
    assert "background subagent(s) have completed" in fake.runs[1]
    assert any("OMEGA" in str(message.get("content")) for message in state.history)
    assert state.history[-1]["content"].startswith("consolidated:")


@pytest.mark.asyncio
async def test_prompt_without_same_turn_delegation_does_not_rerun(monkeypatch):
    acp_agent, state, fake = _make_prompt_agent(monkeypatch)
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="ordinary turn")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == ["ordinary turn"]


@pytest.mark.asyncio
async def test_prompt_join_timeout_is_bounded_and_injects_pending_note(monkeypatch):
    acp_agent, state, fake = _make_prompt_agent(monkeypatch)
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [{"delegation_id": "deleg_late"}],
    )
    monkeypatch.setattr(
        "tools.async_delegation.join",
        lambda delegation_ids, timeout: {
            "completed": [],
            "pending": list(delegation_ids),
        },
    )

    started = time.monotonic()
    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="bounded turn")],
    )
    elapsed = time.monotonic() - started

    assert response.stop_reason == "end_turn"
    assert elapsed < 0.5
    assert fake.runs == ["bounded turn"]
    assert any(
        "still running; results will arrive shortly" in str(message.get("content"))
        for message in state.history
    )
