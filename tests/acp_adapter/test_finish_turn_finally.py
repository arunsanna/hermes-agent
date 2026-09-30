"""Mid-tail failure delivering the final response must not wedge the session running (#115588).

An exception raised while sending the final-response update (persist/provenance/final-text
sends already succeeded) used to skip the bare is_running=False release, leaving the session
wedged running and queued prompts stranded. The turn's tail must release is_running /
current_prompt_text in a finally and still drain the queue, even when delivery itself fails.
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

import acp
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class _BoomConn:
    """Fails the final-text session_update once (transient mid-tail failure)."""

    def __init__(self) -> None:
        self.calls = 0
        self.request_permission = AsyncMock()

    async def session_update(self, session_id, update):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("mid-tail boom")


@pytest.mark.asyncio
async def test_mid_tail_delivery_failure_releases_and_drains():
    run_calls: list[str] = []

    def _agent_factory():
        agent = MagicMock(name="MockAIAgent")
        agent._required_delegation_launching = False
        agent._has_unconsumed_required_delegations.return_value = False

        def _run_conversation(*, user_message, conversation_history, task_id, persist_user_message):
            run_calls.append(persist_user_message)
            return {"final_response": "done", "messages": []}

        agent.run_conversation.side_effect = _run_conversation
        return agent

    manager = SessionManager(agent_factory=_agent_factory)
    server = HermesACPAgent(session_manager=manager)
    resp = await server.new_session(cwd="/tmp")
    state = manager.get_session(resp.session_id)
    state.agent.model, state.agent.provider = "test-model", "openrouter"
    # Pre-seed a queued follow-up so the tail's finally-drain has something to run.
    state.queued_prompts = ["queued-one"]

    conn = _BoomConn()
    server._conn = conn
    server._send_usage_update = AsyncMock()

    with pytest.raises(RuntimeError, match="mid-tail boom"):
        await asyncio.wait_for(
            server.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id),
            timeout=3,
        )

    assert state.is_running is False
    assert state.current_prompt_text == ""
    assert state.queued_prompts == []
    # The drained queued prompt reached the agent as a second turn.
    assert run_calls == ["hi", "queued-one"]
