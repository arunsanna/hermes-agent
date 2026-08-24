"""Turn time-budget check-in: after N minutes of one turn, nudge the model
to pause, summarize progress, and ask the user whether to continue.

The nudge rides the steer piggyback mechanism (appended to the last
tool-role message) so provider role alternation is preserved, and the
model's check-in reply is a normal final message — no new wire vocabulary.
The user replies (e.g. "continue") to resume as an ordinary next turn.
"""

import time
from types import SimpleNamespace

from agent import conversation_loop as loop_mod
from agent.conversation_loop import _maybe_request_turn_checkin
from agent.agent_runtime_helpers import note_turn_start


def make_agent(started_secs_ago, fired=False, minutes=15.0):
    return SimpleNamespace(
        _inflight_turn_started=time.time() - started_secs_ago,
        _turn_checkin_fired=fired,
        _turn_checkin_minutes_override=minutes,
    )


def tool_tail_messages():
    return [
        {"role": "user", "content": "find me a bakery"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "results..."},
    ]


def patch_minutes(monkeypatch):
    monkeypatch.setattr(
        loop_mod,
        "_turn_checkin_minutes",
        lambda agent: float(
            getattr(agent, "_turn_checkin_minutes_override", 0.0)
        ),
    )


def test_checkin_fires_after_budget(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60)
    messages = tool_tail_messages()

    _maybe_request_turn_checkin(agent, messages)

    assert agent._turn_checkin_fired is True
    assert "TURN TIME BUDGET" in messages[-1]["content"]
    assert "ask the user whether to continue" in messages[-1]["content"]


def test_checkin_does_not_fire_before_budget(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=60)
    messages = tool_tail_messages()

    _maybe_request_turn_checkin(agent, messages)

    assert agent._turn_checkin_fired is False
    assert "TURN TIME BUDGET" not in messages[-1]["content"]


def test_checkin_fires_only_once(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60)
    messages = tool_tail_messages()

    _maybe_request_turn_checkin(agent, messages)
    before = messages[-1]["content"]
    _maybe_request_turn_checkin(agent, messages)

    assert messages[-1]["content"] == before


def test_checkin_disabled_when_minutes_zero(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60, minutes=0.0)
    messages = tool_tail_messages()

    _maybe_request_turn_checkin(agent, messages)

    assert agent._turn_checkin_fired is False


def test_checkin_defers_to_explicit_run_budget(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60)
    agent.run_budget_seconds = 900
    messages = tool_tail_messages()

    _maybe_request_turn_checkin(agent, messages)

    assert agent._turn_checkin_fired is False
    assert "TURN TIME BUDGET" not in messages[-1]["content"]


def test_checkin_waits_for_a_tool_message(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60)
    messages = [{"role": "user", "content": "hi"}]

    _maybe_request_turn_checkin(agent, messages)

    # No tool message to piggyback on: stays unfired so a later
    # iteration (once tools have run) can still deliver the nudge.
    assert agent._turn_checkin_fired is False


def test_checkin_appends_to_block_content(monkeypatch):
    patch_minutes(monkeypatch)
    agent = make_agent(started_secs_ago=16 * 60)
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [{"type": "text", "text": "results"}],
        },
    ]

    _maybe_request_turn_checkin(agent, messages)

    assert agent._turn_checkin_fired is True
    assert any(
        "TURN TIME BUDGET" in b.get("text", "")
        for b in messages[-1]["content"]
        if isinstance(b, dict)
    )


def test_note_turn_start_resets_checkin_flag():
    agent = SimpleNamespace(_turn_checkin_fired=True)
    note_turn_start(agent, "turn-1")
    assert agent._turn_checkin_fired is False
