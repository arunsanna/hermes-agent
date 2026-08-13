"""Provider-call failure surfacing: a turn whose terminal result carries
``failed=True`` (retry exhaustion, billing wall, thinking timeout, ...) must
end the ``session/prompt`` request with a JSON-RPC error instead of delivering
the failure text as an ordinary assistant message — without wedging the
session or misattributing errors raised by drained follow-up prompts.
"""

import pytest
import acp
from acp.schema import TextContentBlock

from tests.acp_adapter.test_stop_p0 import (
    _CaptureConn,
    _FakeAgent,
    _make_prompt_agent,
)

FAILURE_TEXT = "API call failed after 3 retries: rate limited"


class _FailingAgent(_FakeAgent):
    """FakeAgent whose Nth run_conversation call returns a failed result."""

    def __init__(self, fail_on_runs=frozenset({1})):
        super().__init__()
        self._fail_on_runs = set(fail_on_runs)

    def run_conversation(
        self, *, user_message, conversation_history, task_id, **_kwargs
    ):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        if len(self.runs) in self._fail_on_runs:
            messages.append({"role": "assistant", "content": FAILURE_TEXT})
            return {
                "final_response": FAILURE_TEXT,
                "messages": messages,
                "completed": False,
                "failed": True,
                "error": "rate limited",
                "failure_reason": "rate_limit",
            }
        final = f"consolidated: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


def _make_failing_prompt_agent(monkeypatch, *, fail_on_runs=frozenset({1})):
    acp_agent, state, _fake, conn = _make_prompt_agent(monkeypatch)
    failing = _FailingAgent(fail_on_runs=fail_on_runs)
    state.agent = failing
    return acp_agent, state, failing, conn


def _agent_message_texts(conn: _CaptureConn) -> list[str]:
    texts = []
    for _session_id, update in conn.updates:
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        kind = getattr(update, "session_update", "")
        if text is not None and "agent_message" in str(kind):
            texts.append(text)
    return texts


@pytest.mark.asyncio
async def test_provider_retry_exhaustion_raises_request_error(monkeypatch):
    acp_agent, state, _failing, _conn = _make_failing_prompt_agent(monkeypatch)

    with pytest.raises(acp.RequestError) as exc_info:
        await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="summarize the meeting")],
        )

    assert exc_info.value.code == -32001
    assert FAILURE_TEXT in str(exc_info.value)
    assert exc_info.value.data["failureReason"] == "rate_limit"
    # The failure text must not survive as durable assistant history: the next
    # turn would feed it back to the model as a genuine prior answer, and a
    # session reload would render it as an ordinary chat bubble.
    assert not any(
        message.get("role") == "assistant"
        and FAILURE_TEXT in str(message.get("content", ""))
        for message in state.history
    ), "failed-turn assistant candidate must be sanitized out of history"


@pytest.mark.asyncio
async def test_provider_call_failure_suppresses_normal_chat_delivery(monkeypatch):
    acp_agent, state, _failing, conn = _make_failing_prompt_agent(monkeypatch)

    with pytest.raises(acp.RequestError):
        await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="summarize the meeting")],
        )

    assert not any(
        FAILURE_TEXT in text for text in _agent_message_texts(conn)
    ), "failure text must not be delivered as an ordinary assistant message"


@pytest.mark.asyncio
async def test_provider_call_failure_does_not_wedge_session(monkeypatch):
    acp_agent, state, failing, _conn = _make_failing_prompt_agent(monkeypatch)

    with pytest.raises(acp.RequestError):
        await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="first ask")],
        )

    assert state.is_running is False
    assert state.current_prompt_text == ""

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="second ask")],
    )
    assert response.stop_reason == "end_turn"
    assert failing.runs == ["first ask", "second ask"]


@pytest.mark.asyncio
async def test_provider_call_failure_drains_queued_follow_up(monkeypatch):
    # Turn 1 succeeds; the queued follow-up (turn 2) fails. The outer prompt
    # must NOT raise (its own turn succeeded) and the drained follow-up's
    # failure must be surfaced via session_update instead of propagating.
    acp_agent, state, failing, conn = _make_failing_prompt_agent(
        monkeypatch, fail_on_runs=frozenset({2})
    )
    state.queued_prompts.append("queued failing ask")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="first ask")],
    )

    assert response.stop_reason == "end_turn"
    assert failing.runs == ["first ask", "queued failing ask"]
    assert state.queued_prompts == []
    assert state.is_running is False
    assert any(
        FAILURE_TEXT in text for text in _agent_message_texts(conn)
    ), "drained follow-up failure should surface as a session_update notice"
