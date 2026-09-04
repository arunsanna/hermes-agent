"""The post-turn background review must never fire for Switchboard sessions.

Switchboard marks its hermes-acp children with HERMES_SESSION_PLATFORM=
"switchboard" in the process env. The ACP adapter binds the session
ContextVar to "" (acp_adapter/server.py -> set_session_vars with no
platform), which suppresses get_session_env's os.environ fallback — so
the gate in AIAgent._spawn_background_review must consult the process
env directly when the ContextVar answer is empty.

Regression for the 2026-07-26 stale-echo incident: the review fork
replayed a 320k-token conversation between turns, ballooned the session
to context overflow, and its stranded output was echoed to the next
user prompt.
"""

from __future__ import annotations

import contextvars

import agent.background_review as background_review_module
import run_agent as run_agent_module
from gateway.session_context import set_session_vars
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "test-session"
    return agent


class _SpyThread:
    started = 0

    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        _SpyThread.started += 1


def _arm(monkeypatch):
    """Patch thread + spawn internals; return the spawn-call recorder."""
    _SpyThread.started = 0
    calls = []

    def _fake_spawn(
        agent, messages_snapshot, review_memory=False, review_skills=False, **kwargs
    ):
        calls.append((review_memory, review_skills))
        return (lambda: None), "prompt"

    monkeypatch.setattr(
        background_review_module, "spawn_background_review_thread", _fake_spawn
    )
    monkeypatch.setattr(run_agent_module.threading, "Thread", _SpyThread)
    return calls


def test_switchboard_env_suppresses_review(monkeypatch):
    calls = _arm(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "switchboard")

    def _in_acp_context():
        # ACP adapter binds the platform ContextVar to "" — the gate must
        # still see the process env through the empty binding.
        set_session_vars(session_key="sess-1")
        _bare_agent()._spawn_background_review(
            messages_snapshot=[], review_memory=True, review_skills=True
        )

    contextvars.copy_context().run(_in_acp_context)
    assert calls == []
    assert _SpyThread.started == 0


def test_switchboard_env_case_insensitive(monkeypatch):
    calls = _arm(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "Switchboard")
    _bare_agent()._spawn_background_review(
        messages_snapshot=[], review_memory=True, review_skills=False
    )
    assert calls == []
    assert _SpyThread.started == 0


def test_gateway_platform_binding_wins_over_env(monkeypatch):
    """A real gateway platform bound in context keeps its review even if a
    stale switchboard value lingers in the process env."""
    calls = _arm(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "switchboard")

    def _in_gateway_context():
        set_session_vars(session_key="sess-2", platform="telegram")
        _bare_agent()._spawn_background_review(
            messages_snapshot=[], review_memory=True, review_skills=False
        )

    contextvars.copy_context().run(_in_gateway_context)
    assert calls == [(True, False)]
    assert _SpyThread.started == 1


def test_review_runs_when_platform_unset(monkeypatch):
    calls = _arm(monkeypatch)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    _bare_agent()._spawn_background_review(
        messages_snapshot=[], review_memory=False, review_skills=True
    )
    assert calls == [(False, True)]
    assert _SpyThread.started == 1
