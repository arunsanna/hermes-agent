"""In-turn ACP keepalive: feed the gateway stall watchdog during silent work.

The gateway force-closes a turn after HERMES_TURN_STALL_SECS (default 300s)
without a single ``session/update``, but healthy operations are wire-silent
far longer: blocking LLM calls (reasoning TTFB floors reach 600s), MCP tools
(300s default), the compression summarizer (300s floor), long single tool
runs. The keepalive loop emits a bounded-rate ``usage_update`` while the
agent still shows liveness, and deliberately goes quiet once the agent has
been silent past the max-silent ceiling so true wedges are still reclaimed
by the gateway watchdog.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from acp_adapter import server as server_mod
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


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

    def save_session(self, session_id):
        return True


def make_server_with_running_state(last_activity_ts):
    manager = NoSaveSessionManager()
    state = SessionState(
        session_id="sess-ka",
        agent=SimpleNamespace(_last_activity_ts=last_activity_ts),
    )
    state.is_running = True
    manager._sessions["sess-ka"] = state
    server = HermesACPAgent(session_manager=manager)
    emissions = []

    async def counting_usage_update(st):
        emissions.append(st.session_id)

    server._send_usage_update = counting_usage_update
    return server, state, emissions


@pytest.mark.asyncio
async def test_keepalive_emits_while_agent_recently_active(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_turn_keepalive_settings", lambda: (0.02, 1800.0)
    )
    server, state, emissions = make_server_with_running_state(time.time())

    task = asyncio.create_task(server._turn_keepalive_loop(state))
    await asyncio.sleep(0.1)
    state.is_running = False
    await asyncio.wait_for(task, timeout=1.0)

    assert len(emissions) >= 2


@pytest.mark.asyncio
async def test_keepalive_goes_quiet_when_agent_silent_past_ceiling(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_turn_keepalive_settings", lambda: (0.02, 1800.0)
    )
    stale = time.time() - 4000.0
    server, state, emissions = make_server_with_running_state(stale)

    task = asyncio.create_task(server._turn_keepalive_loop(state))
    await asyncio.sleep(0.1)
    state.is_running = False
    await asyncio.wait_for(task, timeout=1.0)

    assert emissions == []


@pytest.mark.asyncio
async def test_keepalive_exits_when_turn_ends(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_turn_keepalive_settings", lambda: (0.02, 1800.0)
    )
    server, state, emissions = make_server_with_running_state(time.time())
    state.is_running = False

    task = asyncio.create_task(server._turn_keepalive_loop(state))
    await asyncio.wait_for(task, timeout=1.0)

    assert emissions == []


@pytest.mark.asyncio
async def test_keepalive_cancel_is_clean(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_turn_keepalive_settings", lambda: (0.02, 1800.0)
    )
    server, state, _ = make_server_with_running_state(time.time())

    task = asyncio.create_task(server._turn_keepalive_loop(state))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_keepalive_settings_defaults():
    interval, max_silent = server_mod._turn_keepalive_settings()
    assert interval == pytest.approx(45.0)
    assert max_silent == pytest.approx(1800.0)
