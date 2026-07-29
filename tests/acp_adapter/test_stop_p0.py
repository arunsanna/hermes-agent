"""STOP correctness (Phase 0): guaranteed is_running reset + queue drain,
and a cancel guard around the same-turn delegation join barrier.

See docs/plans/stop-p0-brief.md for the full rationale and verified anchors.
"""

import asyncio
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.events import make_tool_progress_cb
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
        self._required_delegation_launching = False
        # Simulates a STOP arriving from the client WHILE this turn is
        # in-flight on the executor thread (the real race: cancel() sets
        # state.cancel_event concurrently, then the executor result comes
        # back and prompt() checks it before the barrier).
        self._cancel_event_to_set = cancel_event_to_set

    def _has_unconsumed_required_delegations(self):
        return False

    def _finish_acp_provisional_stream(self, *, discard):
        return None

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
    def __init__(self):
        self.replaced_messages = []

    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None

    def replace_messages(self, _session_id, messages, **_kwargs):
        self.replaced_messages.append(list(messages))
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


@pytest.mark.asyncio
async def test_same_loop_cancel_does_not_block_per_required_child():
    manager = SessionManager(agent_factory=lambda **_kwargs: None, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    class _OverlappingConn(_CaptureConn):
        def __init__(self):
            super().__init__()
            self.block_next = False
            self.block_entered = __import__("threading").Event()
            self.block_release = asyncio.Event()

        async def session_update(self, session_id, update):
            if self.block_next:
                self.block_next = False
                self.block_entered.set()
                await self.block_release.wait()
            await super().session_update(session_id, update)

    conn = _OverlappingConn()
    acp_agent.on_connect(conn)
    loop = asyncio.get_running_loop()
    progress = make_tool_progress_cb(
        conn,
        state.session_id,
        loop,
        {},
        {},
    )
    child_ids = [f"child-{index}" for index in range(3)]

    # Seed starts from a worker thread, matching normal delegation workers and
    # ensuring each STOP terminal has an existing same-id ACP card.
    for child_id in child_ids:
        await asyncio.to_thread(
            progress,
            "subagent.start",
            None,
            f"work for {child_id}",
            None,
            subagent_id=child_id,
        )
    for _ in range(20):
        if len(conn.updates) == len(child_ids):
            break
        await asyncio.sleep(0)
    assert len(conn.updates) == len(child_ids)

    # Hold an in-flight heartbeat update while its worker owns child_lock.
    # The old STOP path synchronously interrupted on the ACP loop, then
    # deadlocked acquiring that lock while the worker waited for this loop.
    conn.block_next = True
    heartbeat = asyncio.create_task(
        asyncio.to_thread(
            progress,
            "subagent.heartbeat",
            None,
            "still running",
            None,
            subagent_id=child_ids[0],
            meaningful=False,
        )
    )
    assert await asyncio.to_thread(conn.block_entered.wait, 1)

    class _InterruptsChildren:
        def interrupt(self):
            for child_id in child_ids:
                progress(
                    "subagent.complete",
                    None,
                    "stopped",
                    None,
                    subagent_id=child_id,
                    status="cancelled",
                    supervision_status="cancelled",
                    supervision_terminal=True,
                )

    state.agent = _InterruptsChildren()
    loop.call_later(0.02, conn.block_release.set)
    started = loop.time()
    await acp_agent.cancel(state.session_id)
    elapsed = loop.time() - started
    await heartbeat
    for _ in range(20):
        if len(conn.updates) >= len(child_ids) * 2 + 1:
            break
        await asyncio.sleep(0)

    assert elapsed < 0.25
    starts = [
        update
        for _session_id, update in conn.updates
        if update.session_update == "tool_call"
    ]
    terminals = [
        update
        for _session_id, update in conn.updates
        if update.session_update == "tool_call_update"
        and update.status == "failed"
    ]
    assert len(starts) == len(child_ids)
    assert len(terminals) == len(child_ids)
    assert {update.tool_call_id for update in terminals} == {
        update.tool_call_id for update in starts
    }


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
    acp_agent, state, fake, conn = _make_prompt_agent(
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
    assert [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None)
        == "agent_message_chunk"
    ] == []


@pytest.mark.asyncio
async def test_cancel_after_run_before_final_delivery_suppresses_agent_message(
    monkeypatch,
):
    acp_agent, state, fake, conn = _make_prompt_agent(monkeypatch)

    # Tool-card flush runs after _run_agent returned and before the terminal
    # delivery claim. Flip STOP there to deterministically make cancellation
    # the winner without publishing or durably saving stale assistant prose.
    monkeypatch.setattr(
        "acp_adapter.server.flush_open_tool_calls",
        lambda *_args, **_kwargs: state.cancel_event.set(),
    )
    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    assert fake.runs == ["do work"]
    assert response.stop_reason == "cancelled"
    assert [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None)
        == "agent_message_chunk"
    ] == []
    persisted = acp_agent.session_manager._get_db().replaced_messages[-1]
    assert all(
        "consolidated: do work" not in str(message.get("content", ""))
        for message in persisted
    )


@pytest.mark.asyncio
async def test_final_delivery_claim_serializes_in_flight_stop(monkeypatch):
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title",
        lambda *_args, **_kwargs: None,
    )

    class _PausedFinalConn(_CaptureConn):
        def __init__(self):
            super().__init__()
            self.final_started = asyncio.Event()
            self.release_final = asyncio.Event()

        async def session_update(self, session_id, update):
            if (
                getattr(update, "session_update", None)
                == "agent_message_chunk"
            ):
                self.final_started.set()
                await self.release_final.wait()
            await super().session_update(session_id, update)

    conn = _PausedFinalConn()
    acp_agent.on_connect(conn)
    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[
                TextContentBlock(type="text", text="do work")
            ],
        )
    )
    await asyncio.wait_for(conn.final_started.wait(), timeout=1)

    cancel_task = asyncio.create_task(
        acp_agent.cancel(state.session_id)
    )
    done, _pending = await asyncio.wait(
        {cancel_task},
        timeout=0.05,
    )
    assert done == set()

    conn.release_final.set()
    response = await asyncio.wait_for(prompt_task, timeout=1)
    await asyncio.wait_for(cancel_task, timeout=1)

    assert response.stop_reason == "end_turn"
    assert state.cancel_event.is_set() is False
    final_chunks = [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None)
        == "agent_message_chunk"
    ]
    assert len(final_chunks) == 1
    assert final_chunks[0].content.text == "consolidated: do work"


# ---------------------------------------------------------------------------
# P0.3 — a cancelled turn still drains prompts the user queued after STOP.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_turn_still_drains_queued_follow_ups(monkeypatch):
    acp_agent, state, fake, _conn = _make_prompt_agent(
        monkeypatch, cancel_mid_turn=True
    )
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session",
        lambda **_kwargs: 0,
    )
    # Two prompts the user typed after hitting STOP, queued while the
    # cancelled turn was still (cooperatively) finishing on the executor.
    state.queued_prompts.append("follow-up one")
    state.queued_prompts.append("follow-up two")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    assert response.stop_reason == "cancelled"
    assert state.queued_prompts == []
    assert fake.runs == ["do work", "follow-up one", "follow-up two"]
    assert state.is_running is False


# ---------------------------------------------------------------------------
# HOLE 1 — the join barrier only re-checked cancel_event at the TOP of each
# round. STOP landing while blocked in join() (checkpoint a) or between the
# drain and the continuation dispatch (checkpoint b) must still be caught:
# no continuation _run_agent call, and subagents get interrupted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_join_wait_skips_continuation_and_interrupts(
    monkeypatch,
):
    """STOP lands while the barrier is blocked inside join() itself (not at
    the top of the round). The re-check immediately after join() returns
    must catch it and skip the continuation re-run entirely."""
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [{"delegation_id": "deleg_a"}],
    )

    def _join(delegation_ids, timeout):
        # Simulate STOP arriving while this turn was blocked waiting on
        # join() — the cancel_event flips before join() returns.
        state.cancel_event.set()
        return {"completed": list(delegation_ids), "pending": []}

    monkeypatch.setattr("tools.async_delegation.join", _join)

    interrupt_calls = []
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session",
        lambda **kwargs: interrupt_calls.append(kwargs) or 1,
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    # Only the original turn ran; the join()-completed delegation must NOT
    # trigger a "consolidate results" continuation re-run.
    assert fake.runs == ["do work"]
    assert len(interrupt_calls) == 1
    assert interrupt_calls[0].get("session_key") == state.session_id
    assert state.is_running is False
    assert response.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_cancel_after_drain_before_continuation_skips_continuation(
    monkeypatch,
):
    """STOP lands after join() returns completed delegations and the
    completion drain runs, but before the continuation _run_agent call is
    dispatched. The re-check immediately before that dispatch must catch it."""
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [{"delegation_id": "deleg_b"}],
    )
    monkeypatch.setattr(
        "tools.async_delegation.join",
        lambda delegation_ids, timeout: {
            "completed": list(delegation_ids),
            "pending": [],
        },
    )

    def _drain(pr, formatter, session_id, state_arg):
        # Simulate STOP arriving during the (synchronous) completion drain,
        # i.e. after join() returned but before the continuation dispatch.
        state_arg.cancel_event.set()
        return [{"delegation_id": "deleg_b", "summary": "done"}]

    monkeypatch.setattr(acp_agent, "_drain_session_delegation_completions", _drain)

    interrupt_calls = []
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session",
        lambda **kwargs: interrupt_calls.append(kwargs) or 1,
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    # The original turn ran, but the continuation ("Your background
    # subagent(s) have completed...") must NOT run once cancelled.
    assert fake.runs == ["do work"]
    assert len(interrupt_calls) == 1
    assert interrupt_calls[0].get("session_key") == state.session_id
    assert state.is_running is False
    assert response.stop_reason == "cancelled"


# ---------------------------------------------------------------------------
# HOLE 2 — the except block around the INITIAL executor call resets
# is_running but must also drain state.queued_prompts, same as every other
# return path in prompt().
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_executor_exception_drains_queued_prompts(monkeypatch):
    """`_run_agent` catches every `run_conversation` exception internally and
    turns it into a normal result, so the outer `except Exception` around the
    INITIAL executor call (server.py ~L1808) only fires for failures in the
    executor plumbing itself (context copy, executor scheduling, etc.), not
    agent errors. Reproduce that by making `contextvars.copy_context()` blow
    up before `run_in_executor` is even scheduled."""
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom in initial executor scheduling")

    monkeypatch.setattr("acp_adapter.server.contextvars.copy_context", _boom)
    state.queued_prompts.append("queued during initial crash")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="do work")],
    )

    assert response.stop_reason == "end_turn"
    assert state.is_running is False
    assert state.current_prompt_text == ""
    assert state.queued_prompts == []
    # The executor never ran the agent at all — the crash happened before
    # the initial `_run_agent` dispatch.
    assert fake.runs == []


@pytest.mark.asyncio
async def test_stop_wins_initial_executor_scheduling_exception(monkeypatch):
    """A STOP claimed while executor dispatch is pending stays authoritative
    when that scheduling future later raises."""
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
    loop = asyncio.get_running_loop()
    scheduled = asyncio.Event()
    scheduling_future = loop.create_future()

    def _failing_run_in_executor(*_args, **_kwargs):
        scheduled.set()
        return scheduling_future

    monkeypatch.setattr(loop, "run_in_executor", _failing_run_in_executor)
    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="do work")],
        )
    )
    await asyncio.wait_for(scheduled.wait(), timeout=1)
    await acp_agent.cancel(state.session_id)
    scheduling_future.set_exception(
        RuntimeError("boom in executor scheduling")
    )

    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert state.turn_terminal_winner == "cancelled"
    assert state.is_running is False
    assert fake.runs == []
