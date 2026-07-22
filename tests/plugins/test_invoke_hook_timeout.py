"""A hung plugin hook callback must not block the turn indefinitely.

Regression for the "Hermes turn failed with 300s inactivity" incident: a
``post_llm_call`` / ``on_session_end`` / ``transform_llm_output`` callback that
blocks on an unbounded network/daemon read used to freeze the executor thread
that produces the ACP ``session/prompt`` response. Nothing on the agent side
timed that region out, so the gateway's 300s external stall watchdog was the
only thing that ever ended the turn — surfacing as a spurious failure.

``PluginManager.invoke_hook`` now accepts an opt-in ``_timeout_s`` that bounds
each callback: a hung callback is abandoned after the timeout, the loop
continues to the remaining callbacks, and the turn proceeds. Callers that pass
no timeout keep the exact prior (unbounded) behavior — zero regression for the
per-turn ``pre_llm_call`` context-injection path.
"""
from __future__ import annotations

import threading
import time

from hermes_cli.plugins import PluginManager


def test_invoke_hook_timeout_abandons_hung_callback_and_continues():
    mgr = PluginManager()

    release = threading.Event()
    ran_after_hung = []

    def hung_cb(**_kwargs):
        # Simulates an observability/memory hook blocking on a dead daemon.
        release.wait(timeout=10)
        return "hung-should-be-abandoned"

    def fast_cb(**_kwargs):
        ran_after_hung.append(True)
        return "fast"

    # Hung callback registered FIRST, fast callback after it — proves the loop
    # keeps going past the abandoned callback.
    mgr._hooks["post_llm_call"] = [hung_cb, fast_cb]

    try:
        start = time.monotonic()
        results = mgr.invoke_hook("post_llm_call", _timeout_s=0.3, session_id="s")
        elapsed = time.monotonic() - start

        # Bounded: returned well before the 10s hung sleep.
        assert elapsed < 3.0, f"invoke_hook blocked {elapsed:.1f}s on a hung callback"
        # The callback registered after the hung one still ran.
        assert ran_after_hung == [True]
        # Only the fast callback's return survived; the hung one was skipped.
        assert results == ["fast"]
    finally:
        release.set()


def test_invoke_hook_without_timeout_preserves_behavior():
    mgr = PluginManager()

    def cb(**_kwargs):
        return "ok"

    mgr._hooks["on_session_end"] = [cb]
    # No _timeout_s => unchanged path, callback result collected as before.
    assert mgr.invoke_hook("on_session_end", session_id="s") == ["ok"]
