"""ACP required-delegation integrity and legacy join-barrier coverage."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

import run_agent
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.process_registry import process_registry


class _FakeAgent:
    def __init__(self, emit_dispatch=False):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.runs = []
        self.emit_dispatch = emit_dispatch

    def run_conversation(
        self, *, user_message, conversation_history, task_id, **_kwargs
    ):
        self.runs.append(user_message)
        if self.emit_dispatch and len(self.runs) == 1:
            self.tool_progress_callback(
                "tool.started",
                "delegate_task",
                "review the change",
                {"goal": "review the change", "background": True},
            )
            self.step_callback(
                1,
                [
                    {
                        "name": "delegate_task",
                        "result": json.dumps(
                            {
                                "status": "dispatched",
                                "mode": "background",
                                "delegation_id": "deleg_same_turn",
                            }
                        ),
                    }
                ],
            )
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"consolidated: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class _BlockingRequiredAgent(_FakeAgent):
    """Exercise the real model dispatch policy from an ACP prompt."""

    def __init__(self, child_release):
        super().__init__()
        self.platform = "acp"
        self._delegate_depth = 0
        self.child_release = child_release
        self.interrupted = False
        self.interrupt_calls = 0

    def interrupt(self):
        self.interrupt_calls += 1
        self.interrupted = True
        self.child_release.set()

    def run_conversation(
        self, *, user_message, conversation_history, task_id, **_kwargs
    ):
        self.runs.append(user_message)
        tool_result = run_agent.AIAgent._dispatch_delegate_task(
            self,
            {"goal": "review the answer"},
        )
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        if self.interrupted:
            return {
                "final_response": "",
                "messages": messages,
                "interrupted": True,
            }
        final = (
            "FINAL_WITH_OMEGA"
            if "OMEGA" in tool_result
            else "FINAL_MISSING_CHILD"
        )
        self.stream_delta_callback(final)
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


def _make_prompt_agent(monkeypatch, *, emit_dispatch=False, connect=False):
    fake = _FakeAgent(emit_dispatch=emit_dispatch)
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
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
    return acp_agent, state, fake, conn


def _make_blocking_required_prompt_agent(monkeypatch, child_release):
    fake = _BlockingRequiredAgent(child_release)
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = _CaptureConn()
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


def _agent_message_texts(conn):
    return [
        update.content.text
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None) == "agent_message_chunk"
    ]


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
async def test_required_acp_delegation_blocks_final_until_child_returns(
    monkeypatch,
):
    child_entered = threading.Event()
    child_release = threading.Event()
    background_values = []

    def _delegate(**kwargs):
        background_values.append(kwargs["background"])
        child_entered.set()
        if kwargs["background"]:
            return json.dumps(
                {
                    "status": "dispatched",
                    "mode": "background",
                    "delegation_id": "deleg_premature",
                }
            )
        if not child_release.wait(timeout=5):
            return json.dumps({"results": [{"status": "failed"}]})
        return json.dumps(
            {"results": [{"status": "completed", "summary": "OMEGA"}]}
        )

    monkeypatch.setattr("tools.delegate_tool.delegate_task", _delegate)
    acp_agent, state, _fake, conn = _make_blocking_required_prompt_agent(
        monkeypatch,
        child_release,
    )
    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="give the verified answer")],
        )
    )
    try:
        assert await asyncio.to_thread(child_entered.wait, 2)
        assert background_values == [False]
        assert prompt_task.done() is False
        assert _agent_message_texts(conn) == []
    finally:
        child_release.set()

    response = await asyncio.wait_for(prompt_task, timeout=2)
    await asyncio.sleep(0)

    assert response.stop_reason == "end_turn"
    assert _agent_message_texts(conn) == ["FINAL_WITH_OMEGA"]
    assert state.history[-1]["content"] == "FINAL_WITH_OMEGA"
    assert state.is_running is False


@pytest.mark.asyncio
async def test_cancel_interrupts_required_acp_delegation_without_false_final(
    monkeypatch,
):
    child_entered = threading.Event()
    child_release = threading.Event()
    background_values = []

    def _delegate(**kwargs):
        background_values.append(kwargs["background"])
        child_entered.set()
        child_release.wait(timeout=5)
        return json.dumps(
            {"results": [{"status": "completed", "summary": "OMEGA"}]}
        )

    monkeypatch.setattr("tools.delegate_tool.delegate_task", _delegate)
    acp_agent, state, fake, conn = _make_blocking_required_prompt_agent(
        monkeypatch,
        child_release,
    )
    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="cancel this work")],
        )
    )
    try:
        assert await asyncio.to_thread(child_entered.wait, 2)
        assert background_values == [False]
        await acp_agent.cancel(state.session_id)
        response = await asyncio.wait_for(prompt_task, timeout=2)
    finally:
        child_release.set()

    await asyncio.sleep(0)
    assert response.stop_reason == "cancelled"
    assert fake.interrupt_calls == 1
    assert _agent_message_texts(conn) == []
    assert state.is_running is False


@pytest.mark.asyncio
async def test_prompt_reruns_agent_to_consolidate_same_turn_delegation(
    monkeypatch,
):
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
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
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
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
    acp_agent, state, fake, _conn = _make_prompt_agent(monkeypatch)
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


def _patch_join(monkeypatch, scans, joined):
    scan_iter = iter(scans)
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: next(scan_iter),
    )
    monkeypatch.setattr(
        "tools.async_delegation.join",
        lambda delegation_ids, timeout: joined,
    )


def _dispatch_frames(conn):
    return [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None) in {"tool_call", "tool_call_update"}
    ]


@pytest.mark.asyncio
async def test_joined_completion_updates_original_dispatch_card(monkeypatch):
    acp_agent, state, _fake, conn = _make_prompt_agent(
        monkeypatch, emit_dispatch=True, connect=True
    )
    process_registry.completion_queue.put(_completion_event(state.session_id))
    _patch_join(
        monkeypatch,
        scans=[[{"delegation_id": "deleg_same_turn"}], []],
        joined={"completed": ["deleg_same_turn"], "pending": []},
    )

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="dispatch a reviewer")],
    )

    frames = _dispatch_frames(conn)
    starts = [frame for frame in frames if frame.session_update == "tool_call"]
    dispatch_updates = [
        frame
        for frame in frames
        if frame.session_update == "tool_call_update"
        and frame.tool_call_id == starts[0].tool_call_id
    ]
    assert len(starts) == 1
    assert [update.status for update in dispatch_updates] == [
        "in_progress",
        "completed",
    ]
    assert "OMEGA" in "".join(str(part) for part in dispatch_updates[-1].content)


@pytest.mark.asyncio
async def test_timeout_flush_leaves_no_in_progress_dispatch_card(monkeypatch):
    acp_agent, state, _fake, conn = _make_prompt_agent(
        monkeypatch, emit_dispatch=True, connect=True
    )
    _patch_join(
        monkeypatch,
        scans=[[{"delegation_id": "deleg_same_turn"}]],
        joined={"completed": [], "pending": ["deleg_same_turn"]},
    )

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="dispatch a reviewer")],
    )

    frames = _dispatch_frames(conn)
    dispatch_id = next(
        frame.tool_call_id for frame in frames if frame.session_update == "tool_call"
    )
    updates = [
        frame
        for frame in frames
        if frame.session_update == "tool_call_update"
        and frame.tool_call_id == dispatch_id
    ]
    assert updates[-1].status != "in_progress"
    assert "later turn" in "".join(str(part) for part in updates[-1].content)


@pytest.mark.asyncio
async def test_joined_dispatch_without_result_event_is_failed(monkeypatch):
    acp_agent, state, _fake, conn = _make_prompt_agent(
        monkeypatch, emit_dispatch=True, connect=True
    )
    _patch_join(
        monkeypatch,
        scans=[[{"delegation_id": "deleg_same_turn"}]],
        joined={"completed": ["deleg_same_turn"], "pending": []},
    )

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="dispatch a reviewer")],
    )

    frames = _dispatch_frames(conn)
    dispatch_id = next(
        frame.tool_call_id for frame in frames if frame.session_update == "tool_call"
    )
    updates = [
        frame
        for frame in frames
        if frame.session_update == "tool_call_update"
        and frame.tool_call_id == dispatch_id
    ]
    assert updates[-1].status == "failed"
    assert "subagent result not received" in "".join(
        str(part) for part in updates[-1].content
    )


@pytest.mark.asyncio
async def test_executor_error_flushes_dispatch_card_failed(monkeypatch):
    acp_agent, state, _fake, conn = _make_prompt_agent(
        monkeypatch, emit_dispatch=True, connect=True
    )

    class _RunThenRaiseExecutor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            def run_then_raise():
                fn(*args, **kwargs)
                raise RuntimeError("executor boundary failed")

            return super().submit(run_then_raise)

    with _RunThenRaiseExecutor(max_workers=1) as executor:
        monkeypatch.setattr("acp_adapter.server._executor", executor)
        response = await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="dispatch a reviewer")],
        )

    assert response.stop_reason == "end_turn"
    frames = _dispatch_frames(conn)
    dispatch_id = next(
        frame.tool_call_id for frame in frames if frame.session_update == "tool_call"
    )
    updates = [
        frame
        for frame in frames
        if frame.session_update == "tool_call_update"
        and frame.tool_call_id == dispatch_id
    ]
    assert updates[-1].status == "failed"
    assert "subagent result not received" in "".join(
        str(part) for part in updates[-1].content
    )
