"""A mid-tail transport failure releases the real prompt lifecycle and drains its queue."""

from types import SimpleNamespace

import pytest

from acp.schema import TextContentBlock
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


@pytest.mark.asyncio
async def test_finish_turn_mid_tail_failure_releases_and_drains(monkeypatch):
    runs = []

    def run_conversation(**kwargs):
        runs.append(kwargs["user_message"])
        return {"final_response": "done", "messages": []}

    agent = SimpleNamespace(
        session_id="h1", model="m", run_conversation=run_conversation,
        _required_delegation_launching=False,
        _has_unconsumed_required_delegations=lambda: False,
    )
    manager = SessionManager(agent_factory=lambda: agent)
    server = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    monkeypatch.setattr(manager, "save_session", lambda sid: None)
    monkeypatch.setattr(server, "_ensure_delegation_watcher", lambda loop: None)
    monkeypatch.setattr("tools.async_delegation.running_for_session", lambda *a: [])

    class BoomConn:
        calls = 0

        async def request_permission(self, *args, **kwargs):
            raise AssertionError("no tool approval expected")

        async def session_update(self, session_id, update):
            self.calls += 1
            if self.calls == 1:
                # A prompt accepted during final delivery must drain even if delivery fails.
                state.queued_prompts.append("queued-one")
                raise RuntimeError("mid-tail boom")

    server._conn = BoomConn()
    with pytest.raises(RuntimeError, match="mid-tail boom"):
        await server.prompt([TextContentBlock(type="text", text="hello")], state.session_id)

    assert state.is_running is False
    assert state.current_prompt_text == ""
    assert state.queued_prompts == []
    assert runs == ["hello", "queued-one"]
