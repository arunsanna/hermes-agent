"""ACP required-delegation integrity and legacy join-barrier coverage."""

import asyncio
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

import run_agent
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import (
    SessionManager,
    UnsafeSessionTranscriptError,
)
from tools.process_registry import process_registry


class _FakeAgent:
    def __init__(self, emit_dispatch=False, required_failure=False):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.runs = []
        self.emit_dispatch = emit_dispatch
        self.required_failure = required_failure
        self._required_delegation_launching = False

    def _has_unconsumed_required_delegations(self):
        return False

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
        if self.required_failure:
            return {
                "final_response": "MUST NOT LEAK REQUIRED CANDIDATE",
                "messages": messages,
                "completed": False,
                "failed": True,
                "error": "required_delegation_observation_failed",
                "required_delegation_pending": True,
            }
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
        dispatch = json.loads(tool_result)
        if (
            dispatch.get("status") == "dispatched"
            and dispatch.get("mode") == "required"
        ):
            # The real conversation loop waits on the required controller
            # after the background launch. This ACP-adapter fake models that
            # barrier without duplicating the controller integration covered
            # by the conversation-loop tests.
            self.child_release.wait(timeout=5)
            tool_result = json.dumps(
                {"results": [{"status": "completed", "summary": "OMEGA"}]}
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


class _DurableDb:
    def __init__(self):
        self.sessions = {}
        self.messages = {}
        self.message_loads = 0

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def create_session(
        self, *, session_id, source, model=None, model_config=None, **_kwargs
    ):
        self.sessions[session_id] = {
            "id": session_id,
            "source": source,
            "model": model,
            "model_config": json.dumps(model_config or {}),
        }

    def update_session_meta(self, session_id, model_config, model=None):
        self.sessions[session_id]["model_config"] = model_config
        self.sessions[session_id]["model"] = model

    def replace_messages(self, session_id, messages, active_only=False):
        self.messages[session_id] = copy.deepcopy(messages)

    def get_messages_as_conversation(
        self, session_id, repair_alternation=False
    ):
        self.message_loads += 1
        return copy.deepcopy(self.messages.get(session_id, []))

    def has_archived_messages(self, _session_id):
        return False


class _FailingCorrectionDb(_DurableDb):
    def __init__(self):
        super().__init__()
        self.correction_failures = 0
        self.correction_active_only = []
        self.archived_messages = [{"role": "user", "content": "archived"}]
        self.metadata_failures = 0

    def update_session_meta(self, session_id, model_config, model=None):
        if self.metadata_failures:
            self.metadata_failures -= 1
            raise RuntimeError("simulated metadata failure")
        super().update_session_meta(session_id, model_config, model)

    def replace_messages(self, session_id, messages, active_only=False):
        self.correction_active_only.append(active_only)
        if self.correction_failures:
            self.correction_failures -= 1
            raise RuntimeError("simulated correction failure")
        if not active_only:
            self.archived_messages = []
        super().replace_messages(
            session_id,
            messages,
            active_only=active_only,
        )


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


def _make_prompt_agent(
    monkeypatch,
    *,
    emit_dispatch=False,
    connect=False,
    required_failure=False,
):
    fake = _FakeAgent(
        emit_dispatch=emit_dispatch,
        required_failure=required_failure,
    )
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
                    "mode": "required",
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
        assert background_values == [True]
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
        return json.dumps(
            {
                "status": "dispatched",
                "mode": "required",
                "delegation_id": "deleg_cancel",
            }
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
        assert background_values == [True]
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
async def test_required_observation_failure_emits_failed_tool_and_no_answer(
    monkeypatch,
):
    acp_agent, state, _fake, conn = _make_prompt_agent(
        monkeypatch,
        connect=True,
        required_failure=True,
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="required work")],
    )

    assert response.stop_reason == "refusal"
    assert _agent_message_texts(conn) == []
    failure_frames = [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None)
        in {"tool_call", "tool_call_update"}
    ]
    assert [frame.session_update for frame in failure_frames] == [
        "tool_call",
        "tool_call_update",
    ]
    assert failure_frames[1].tool_call_id == failure_frames[0].tool_call_id
    assert failure_frames[1].status == "failed"
    assert "could not safely persist" in "".join(
        str(part) for part in failure_frames[1].content
    ).lower()
    assert state.history[-1]["role"] == "user"
    assert state.is_running is False


@pytest.mark.asyncio
async def test_escaped_required_candidate_is_removed_from_durable_reload(
    monkeypatch,
):
    db = _DurableDb()

    class _EscapedRequiredAgent(_FakeAgent):
        def __init__(self, session_id):
            super().__init__()
            self.session_id = session_id
            self.platform = "acp"
            self._session_db = db
            self._session_db_created = True
            self._required_delegation_launching = True

        def _has_unconsumed_required_delegations(self):
            return True

        def _finish_acp_provisional_stream(self, *, discard):
            assert discard is True

        def run_conversation(
            self, *, user_message, conversation_history, **_kwargs
        ):
            messages = list(conversation_history or [])
            messages.append({"role": "user", "content": user_message})
            messages.append(
                {
                    "role": "assistant",
                    "content": "STALE REQUIRED CANDIDATE",
                    "codex_message_items": [
                        {
                            "type": "message",
                            "content": "STALE REQUIRED CANDIDATE",
                        }
                    ],
                    "codex_reasoning_items": [
                        {"type": "reasoning", "id": "reason-1"}
                    ],
                    "tool_calls": [
                        {
                            "id": "delegate-1",
                            "type": "function",
                            "function": {
                                "name": "delegate_task",
                                "arguments": "{}",
                            },
                        }
                    ],
                    "anthropic_content_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "signed continuity",
                            "signature": "sig",
                        },
                        {
                            "type": "text",
                            "text": "STALE REQUIRED CANDIDATE",
                        },
                        {
                            "type": "tool_use",
                            "id": "delegate-1",
                            "name": "delegate_task",
                            "input": {},
                        },
                    ],
                }
            )
            # Model a direct/fatal path that flushed before the final ACP
            # integrity check.
            db.replace_messages(self.session_id, messages)
            return {
                "final_response": "STALE REQUIRED CANDIDATE",
                "messages": messages,
            }

    def _factory():
        return _EscapedRequiredAgent("")

    manager = SessionManager(agent_factory=_factory, db=db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    state.agent.session_id = state.session_id
    conn = _CaptureConn()
    acp_agent.on_connect(conn)
    monkeypatch.setattr(
        acp_agent, "_ensure_delegation_watcher", lambda _loop: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="required work")],
    )

    assert response.stop_reason == "refusal"
    assert _agent_message_texts(conn) == []
    assert "STALE REQUIRED CANDIDATE" not in json.dumps(
        db.messages[state.session_id]
    )
    protocol_message = db.messages[state.session_id][-1]
    assert protocol_message["role"] == "assistant"
    assert protocol_message["content"] == ""
    assert protocol_message["tool_calls"][0]["id"] == "delegate-1"
    assert protocol_message["codex_reasoning_items"] == [
        {"type": "reasoning", "id": "reason-1"}
    ]
    assert [
        block["type"]
        for block in protocol_message["anthropic_content_blocks"]
    ] == ["thinking", "tool_use"]

    restored_manager = SessionManager(
        agent_factory=lambda **_kwargs: _FakeAgent(),
        db=db,
    )
    restored = restored_manager.get_session(state.session_id)
    assert restored is not None
    assert "STALE REQUIRED CANDIDATE" not in json.dumps(restored.history)
    assert restored.history[-1]["tool_calls"][0]["id"] == "delegate-1"


def _build_cancelled_durable_prompt(monkeypatch, db, correction_failures):
    class _CancelledPersistingAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.platform = "acp"
            self.session_id = ""
            self._session_db = db
            self._session_db_created = True
            self.cancel_event = None

        def _finish_acp_provisional_stream(self, *, discard):
            assert discard is True

        def run_conversation(
            self, *, user_message, conversation_history, **_kwargs
        ):
            self.runs.append(user_message)
            messages = list(conversation_history or [])
            messages.append({"role": "user", "content": user_message})
            messages.append(
                {"role": "assistant", "content": "REJECTED STOP CANDIDATE"}
            )
            # Model an agent-owned incremental flush without destructively
            # touching pre-existing archived rows.
            db.messages[self.session_id] = copy.deepcopy(messages)
            db.correction_failures = correction_failures
            self.cancel_event.set()
            return {
                "final_response": "REJECTED STOP CANDIDATE",
                "messages": messages,
            }

    fake = _CancelledPersistingAgent()
    manager = SessionManager(agent_factory=lambda: fake, db=db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    fake.session_id = state.session_id
    fake.cancel_event = state.cancel_event
    conn = _CaptureConn()
    acp_agent.on_connect(conn)
    monkeypatch.setattr(
        acp_agent, "_ensure_delegation_watcher", lambda _loop: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )
    return acp_agent, state, conn


@pytest.mark.asyncio
async def test_stop_transient_rewrite_failure_recovers_durable_reload(
    monkeypatch,
):
    db = _FailingCorrectionDb()
    acp_agent, state, conn = _build_cancelled_durable_prompt(
        monkeypatch,
        db,
        correction_failures=1,
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="cancelled work")],
    )

    assert response.stop_reason == "cancelled"
    assert _agent_message_texts(conn) == []
    assert "REJECTED STOP CANDIDATE" not in json.dumps(
        db.messages[state.session_id]
    )
    assert db.correction_active_only[-2:] == [True, True]
    assert db.archived_messages == [
        {"role": "user", "content": "archived"}
    ]
    marker = json.loads(db.sessions[state.session_id]["model_config"])
    assert "acp_transcript_correction_poisoned" not in marker
    assert state.transcript_correction_poisoned is False
    assert state.transcript_correction_poison_persisted is None

    restored_manager = SessionManager(
        agent_factory=lambda: _FakeAgent(),
        db=db,
    )
    restored = restored_manager.get_session(state.session_id)
    assert restored is not None
    assert "REJECTED STOP CANDIDATE" not in json.dumps(restored.history)


@pytest.mark.asyncio
async def test_stop_rewrite_failure_budget_exhausted_self_heals_on_fresh_restore(
    monkeypatch,
):
    """The in-process retry loop's own 3-attempt budget can genuinely
    exhaust (this in-process behavior is unchanged: the resident owner
    stays poisoned and keeps refusing without a redo — self-heal is scoped
    to the restore path, not to an already-resident session, so it never
    hammers a session that already failed within this same process). But
    the poison marker is a redo flag, not a tombstone: once
    ``_FailingCorrectionDb``'s failure budget is exhausted, the underlying
    write would now genuinely succeed if retried, so a brand-new process
    restoring the same session must self-heal instead of refusing forever.

    (Formerly named ...is_explicit_and_fail_closed, when a fresh restore
    unconditionally rejected the marker with no redo attempt at all — see
    test_crash_between_flush_and_correction_blocks_resume_via_poison_marker
    for the case where the redo itself keeps failing and refusal is still
    correct.)
    """
    db = _FailingCorrectionDb()
    acp_agent, state, conn = _build_cancelled_durable_prompt(
        monkeypatch,
        db,
        correction_failures=3,
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="cancelled work")],
    )

    assert response.stop_reason == "cancelled"
    assert _agent_message_texts(conn) == []
    assert db.correction_active_only[-3:] == [True, True, True]
    assert db.archived_messages == [
        {"role": "user", "content": "archived"}
    ]
    marker = json.loads(
        db.sessions[state.session_id]["model_config"]
    )
    assert marker["acp_transcript_correction_poisoned"] is True
    assert state.transcript_correction_poisoned is True
    assert state.transcript_correction_poison_persisted is True
    failure_updates = [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None) == "tool_call_update"
        and getattr(update, "status", None) == "failed"
    ]
    assert len(failure_updates) == 1
    assert "could not durably" in "".join(
        str(part) for part in failure_updates[0].content
    ).lower()

    # The resident owner must still refuse before invoking the model again
    # — self-heal never runs for an already-in-memory session, only on a
    # fresh restore. Unchanged from before self-heal existed.
    second_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="try again")],
    )
    assert second_response.stop_reason == "refusal"
    assert state.agent.runs == ["cancelled work"]

    # A fresh process restoring the same session redoes the correction
    # instead of refusing outright. _FailingCorrectionDb.correction_failures
    # is now exhausted (all 3 budgeted failures were already consumed
    # above), so replace_messages succeeds this time: the redo strips the
    # rejected candidate for real, durably clears the marker, and resume
    # proceeds with the corrected history.
    for method_name in ("load_session", "resume_session"):
        db.correction_failures = 0
        db.metadata_failures = 0
        factory_calls = []

        def _fresh_factory():
            factory_calls.append(True)
            return _FakeAgent()

        fresh_manager = SessionManager(
            agent_factory=_fresh_factory,
            db=db,
        )
        fresh_agent = HermesACPAgent(session_manager=fresh_manager)
        fresh_conn = _CaptureConn()
        fresh_agent.on_connect(fresh_conn)

        result = await getattr(fresh_agent, method_name)(
            cwd=".",
            session_id=state.session_id,
        )

        assert result is not None
        assert factory_calls == [True]
        healed_marker = json.loads(db.sessions[state.session_id]["model_config"])
        assert "acp_transcript_correction_poisoned" not in healed_marker
        assert "REJECTED STOP CANDIDATE" not in json.dumps(
            db.messages[state.session_id]
        )
        assert _agent_message_texts(fresh_conn) == []

        # Reset back to poisoned for the next method_name iteration so both
        # load_session and resume_session independently exercise the same
        # self-heal-from-poisoned starting state.
        db.sessions[state.session_id]["model_config"] = json.dumps(
            {**healed_marker, "acp_transcript_correction_poisoned": True}
        )
        db.messages[state.session_id] = [
            {"role": "user", "content": "cancelled work"},
            {"role": "assistant", "content": "REJECTED STOP CANDIDATE"},
        ]


@pytest.mark.asyncio
async def test_stop_marker_write_failure_remains_resident_fail_closed(
    monkeypatch,
):
    db = _FailingCorrectionDb()
    db.metadata_failures = 100
    acp_agent, state, conn = _build_cancelled_durable_prompt(
        monkeypatch,
        db,
        correction_failures=3,
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="cancelled work")],
    )

    assert response.stop_reason == "cancelled"
    assert _agent_message_texts(conn) == []
    assert state.transcript_correction_poisoned is True
    assert state.transcript_correction_poison_persisted is False
    marker = json.loads(db.sessions[state.session_id]["model_config"])
    assert "acp_transcript_correction_poisoned" not in marker
    assert db.archived_messages == [
        {"role": "user", "content": "archived"}
    ]

    failure_updates = [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None) == "tool_call_update"
        and getattr(update, "status", None) == "failed"
    ]
    assert len(failure_updates) == 1
    failure_text = "".join(
        str(part) for part in failure_updates[0].content
    ).lower()
    assert "safety marker" in failure_text
    assert "do not resume" in failure_text

    # Durable safety cannot be claimed when the metadata write failed, but the
    # live owner must remain poisoned and refuse without another model call.
    second_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="try again")],
    )
    assert second_response.stop_reason == "refusal"
    assert state.agent.runs == ["cancelled work"]


def test_crash_mid_correction_durably_sets_poison_marker_before_write(
    tmp_path,
):
    """Regression for the crash-window defect: a process kill landing
    between "the tainted candidate was already flushed to state.db" (the
    turn's own incremental persistence, before ``_rewrite_agent_active_
    history`` even runs) and "the corrective ``replace_messages`` call
    actually completes" must still leave a durable trace — the poison
    marker — even though the correction itself never wrote anything.

    (Whether a *subsequent* restore attempt then self-heals or keeps
    refusing is a separate question, covered by
    ``test_stuck_marker_on_already_corrected_session_self_heals_on_restore``
    (marker-only failure — clean transcript, healable) and
    ``test_genuine_taint_that_keeps_failing_still_refuses_on_restore``
    (real taint whose redo also keeps failing — must keep refusing). This
    test isolates only the narrower, original round-2 claim: does the
    proactive, poison-first marker write survive a kill that happens before
    the correction's own write, independent of what a later restore
    attempt does with that marker.)

    This constructs exactly the durable state a real kill would leave, using
    a real SQLite-backed SessionDB (not the in-memory fakes used elsewhere
    in this file), and calls the real ``_rewrite_agent_active_history``
    under test (not a hand-simulated stand-in for it) with a db wrapper
    whose ``replace_messages`` raises ``KeyboardInterrupt`` — a
    ``BaseException``, not caught by any ``except Exception:`` in
    ``_rewrite_agent_active_history`` — the instant correction is
    attempted, before any write happens. This faithfully simulates an
    uncatchable process kill occurring mid-call: unlike the existing
    attempted-and-failed tests (``_FailingCorrectionDb``, an ordinary
    ``Exception`` the retry loop swallows and retries, eventually reaching
    the old mark-on-failure code at the bottom of the function),
    ``replace_messages`` is never actually completed and the function's own
    failure-path marker write at the bottom is also never reached — proving
    whether the DURABLE marker exists at all depends entirely on whether the
    proactive, poison-first write at the top of the function already ran.
    """
    from hermes_state import SessionDB
    from acp_adapter.server import _rewrite_agent_active_history

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        fake = _FakeAgent()
        manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=db)
        state = manager.create_session(cwd=".")
        fake.session_id = state.session_id

        # The turn's own incremental persistence already flushed the
        # rejected candidate, complete with its byte-fidelity api_content
        # sidecar, before the simulated crash.
        leaked_api_content = (
            "STALE REQUIRED CANDIDATE\n\n"
            "<memory-context>\nleaked child result\n</memory-context>\n"
        )
        db.append_message(
            state.session_id,
            "assistant",
            content="STALE REQUIRED CANDIDATE",
            api_content=leaked_api_content,
        )
        messages = db.get_messages_as_conversation(state.session_id)

        class _KillOnReplace:
            """Delegates everything to the real db except replace_messages,
            which simulates the process dying the instant correction is
            attempted -- before the correction writes anything."""

            def __getattr__(self, name):
                return getattr(db, name)

            def replace_messages(self, *args, **kwargs):
                raise KeyboardInterrupt("simulated process kill mid-correction")

        fake._session_db = _KillOnReplace()

        with pytest.raises(KeyboardInterrupt):
            _rewrite_agent_active_history(fake, messages, state, manager)

        # Read the raw DB row directly -- deliberately NOT going through
        # get_session/_restore here, so this test stays isolated to the
        # narrow "did the marker write survive the kill" claim without also
        # exercising the self-heal redo (covered by the dedicated tests
        # above).
        row = db.get_session(state.session_id)
        marker = json.loads(row["model_config"])
        assert marker.get("acp_transcript_correction_poisoned") is True

        # The tainted row (and its api_content sidecar) really is still
        # sitting there uncorrected on the real db — proving it is the
        # durable marker, not a lucky sanitize-on-reload, that closed this
        # window: replace_messages never actually wrote anything.
        raw_reload = db.get_messages_as_conversation(state.session_id)
        assert raw_reload[-1]["content"] == "STALE REQUIRED CANDIDATE"
        assert raw_reload[-1]["api_content"] == leaked_api_content
    finally:
        db.close()


def test_stuck_marker_on_already_corrected_session_self_heals_on_restore(
    tmp_path,
):
    """Required regression (a): the round-3 defect this fix closes.

    ``replace_messages`` genuinely succeeds (the transcript is truly clean)
    but the immediately-following marker CLEAR fails every time it is
    attempted (a real, not budget-exhaustible, persistent failure of just
    the clear write) — so the durable marker is stuck ``True`` even though
    nothing is actually wrong with the transcript. A subsequent restore in
    a brand-new process must self-heal: it redoes sanitize (a no-op on the
    already-clean content) + rewrite (a no-op rewrite of the same clean
    bytes) + clear, and since THIS clear attempt is not blocked, resume
    succeeds with the marker durably cleared and the history intact/clean.
    """
    from hermes_state import SessionDB
    from acp_adapter.server import (
        _rewrite_agent_active_history,
        _sanitize_failed_turn_history,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        fake = _FakeAgent()
        manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=db)
        state = manager.create_session(cwd=".")
        fake.session_id = state.session_id

        db.append_message(state.session_id, "user", content="do the required work")
        db.append_message(
            state.session_id,
            "assistant",
            content="STALE REQUIRED CANDIDATE",
            api_content="STALE REQUIRED CANDIDATE\n\n<memory-context>leak</memory-context>",
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {"name": "delegate_task", "arguments": "{}"},
                }
            ],
        )
        # The real call sites always sanitize BEFORE calling
        # _rewrite_agent_active_history (server.py's own branches, and
        # _self_heal_poisoned_history) -- _rewrite_agent_active_history
        # itself just persists whatever list it is given, verbatim.
        messages = _sanitize_failed_turn_history(
            db.get_messages_as_conversation(state.session_id), baseline_count=1
        )

        real_update_session_meta = db.update_session_meta
        clear_should_fail = True

        def _flaky_update_session_meta(session_id, model_config, model=None):
            nonlocal clear_should_fail
            if clear_should_fail:
                # A clear write omits the poison key entirely (session.py's
                # _persist only ever adds the key when poisoned=True, never
                # writes it as an explicit False) -- so "no poison key in
                # this write" means this IS the clear call. Only that call
                # fails; the proactive mark (which DOES include the key)
                # must still succeed so the marker itself is genuinely
                # durable going into the restore.
                try:
                    parsed = json.loads(model_config or "{}")
                except (TypeError, ValueError):
                    parsed = {}
                if "acp_transcript_correction_poisoned" not in parsed:
                    raise RuntimeError("simulated persistent clear failure")
            return real_update_session_meta(session_id, model_config, model)

        db.update_session_meta = _flaky_update_session_meta
        fake._session_db = db

        rewrite_ok = _rewrite_agent_active_history(fake, messages, state, manager)
        assert rewrite_ok is False  # clear never got through -> reported failed

        row = db.get_session(state.session_id)
        marker = json.loads(row["model_config"])
        assert marker.get("acp_transcript_correction_poisoned") is True
        # The correction itself DID durably land -- this is the "already
        # clean" precondition self-heal must treat as a no-op.
        corrected = db.get_messages_as_conversation(state.session_id)
        assert corrected[-1]["content"] == ""
        assert "api_content" not in corrected[-1]

        # The clear-write failure was a one-time fluke, not indefinite —
        # let it succeed on the next attempt (self-heal's own retry / the
        # restore's redo).
        clear_should_fail = False

        fresh_manager = SessionManager(
            agent_factory=lambda **_kwargs: _FakeAgent(), db=db
        )
        restored = fresh_manager.get_session(state.session_id)  # must NOT raise

        assert restored is not None
        healed_marker = json.loads(db.get_session(state.session_id)["model_config"])
        assert "acp_transcript_correction_poisoned" not in healed_marker
        assert restored.transcript_correction_poisoned is False
        assert restored.history[-1]["content"] == ""
        assert "STALE REQUIRED CANDIDATE" not in json.dumps(restored.history)
    finally:
        db.close()


def test_genuine_taint_that_keeps_failing_still_refuses_on_restore(tmp_path):
    """Required regression (b): no regression of the round-2 guarantee.

    The correction never completes at all — ``replace_messages`` fails
    every single time it is called, indefinitely (not a budget that
    eventually exhausts) — so both the original attempt AND the restore's
    own self-heal redo keep failing. Resume must still refuse: the poison
    marker being a redo flag does not mean it becomes a rubber stamp.
    """
    from hermes_state import SessionDB
    from acp_adapter.server import _rewrite_agent_active_history

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        fake = _FakeAgent()
        manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=db)
        state = manager.create_session(cwd=".")
        fake.session_id = state.session_id

        leaked_api_content = (
            "STALE REQUIRED CANDIDATE\n\n"
            "<memory-context>\nleaked child result\n</memory-context>\n"
        )
        db.append_message(
            state.session_id,
            "assistant",
            content="STALE REQUIRED CANDIDATE",
            api_content=leaked_api_content,
        )
        messages = db.get_messages_as_conversation(state.session_id)

        class _AlwaysFailsReplace:
            """replace_messages fails every time, indefinitely -- unlike
            _FailingCorrectionDb's exhaustible budget, this never recovers,
            simulating genuinely broken/unreachable storage for this
            session's correction specifically."""

            def __getattr__(self, name):
                return getattr(db, name)

            def replace_messages(self, *args, **kwargs):
                raise RuntimeError("simulated persistent correction failure")

        fake._session_db = _AlwaysFailsReplace()

        rewrite_ok = _rewrite_agent_active_history(fake, messages, state, manager)
        assert rewrite_ok is False

        row = db.get_session(state.session_id)
        marker = json.loads(row["model_config"])
        assert marker.get("acp_transcript_correction_poisoned") is True

        # A fresh restore's self-heal redo is subject to the SAME broken
        # storage (same underlying db, same always-failing replace_messages
        # for this session) -- it must keep refusing, not silently resume
        # with the still-tainted candidate.
        fresh_manager = SessionManager(
            agent_factory=lambda **_kwargs: _FakeAgent(), db=_AlwaysFailsReplace()
        )
        with pytest.raises(UnsafeSessionTranscriptError):
            fresh_manager.get_session(state.session_id)
        # Not a one-shot fluke: refuses on every attempt.
        with pytest.raises(UnsafeSessionTranscriptError):
            fresh_manager.get_session(state.session_id)

        raw_reload = db.get_messages_as_conversation(state.session_id)
        assert raw_reload[-1]["content"] == "STALE REQUIRED CANDIDATE"
        assert raw_reload[-1]["api_content"] == leaked_api_content
    finally:
        db.close()


def test_self_heal_on_already_clean_history_is_a_no_op(tmp_path):
    """Required regression (c): idempotency.

    A poisoned session can never accept a new user turn (the poison check
    refuses before one could ever be appended), so in production the
    poisoned marker is ALWAYS correlated with a turn that already needed —
    or already got — correction; it is never spuriously poisoned on top of
    an unrelated, ordinary completed answer. The realistic "already clean"
    shape self-heal must be idempotent on is therefore: a transcript that
    has ALREADY been through one successful sanitize+rewrite pass (tail
    assistant candidate already reduced to content="" with only its
    protocol fields surviving — exactly what
    ``test_stuck_marker_on_already_corrected_session_self_heals_on_restore``
    produces), poisoned a second time for some unrelated reason (e.g. that
    same stuck-clear failure recurring). Self-heal must leave the
    already-sanitized bytes byte-for-byte unchanged — only the marker
    lifecycle moves.
    """
    from hermes_state import SessionDB
    from acp_adapter.server import _rewrite_agent_active_history

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        fake = _FakeAgent()
        manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=db)
        state = manager.create_session(cwd=".")
        fake.session_id = state.session_id

        db.append_message(state.session_id, "user", content="do the required work")
        # Already in POST-sanitize form: empty content, protocol preserved
        # (tool_calls survives _sanitize_required_assistant_candidate, which
        # only clears content/api_content/codex_message_items/text blocks).
        db.append_message(
            state.session_id,
            "assistant",
            content="",
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {"name": "delegate_task", "arguments": "{}"},
                }
            ],
        )
        before = db.get_messages_as_conversation(state.session_id)

        # Directly durably poison the marker a second time (standing in for
        # the stuck-clear failure recurring on an already-corrected
        # session) without touching the already-clean transcript.
        assert manager.mark_transcript_correction_poisoned(state) is True

        fresh_manager = SessionManager(
            agent_factory=lambda **_kwargs: _FakeAgent(), db=db
        )
        restored = fresh_manager.get_session(state.session_id)  # must NOT raise

        assert restored is not None
        assert restored.history == before
        healed_marker = json.loads(db.get_session(state.session_id)["model_config"])
        assert "acp_transcript_correction_poisoned" not in healed_marker
        assert restored.transcript_correction_poisoned is False

        # Calling _rewrite_agent_active_history directly a second time
        # (simulating self-heal running again on an already-healed,
        # already-clean history) is also a pure no-op besides the marker
        # lifecycle -- content is untouched.
        again_ok = _rewrite_agent_active_history(fake, before, state, manager)
        assert again_ok is True
        assert db.get_messages_as_conversation(state.session_id) == before
    finally:
        db.close()


@pytest.mark.asyncio
async def test_required_agent_exception_emits_failed_tool_and_no_answer(
    monkeypatch,
):
    class _ExceptionAgent(_FakeAgent):
        _required_delegation_launching = True

        def _has_unconsumed_required_delegations(self):
            return True

        def run_conversation(self, **_kwargs):
            raise RuntimeError("post-dispatch failure")

    fake = _ExceptionAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = _CaptureConn()
    acp_agent.on_connect(conn)
    monkeypatch.setattr(
        acp_agent, "_ensure_delegation_watcher", lambda _loop: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="required work")],
    )

    assert response.stop_reason == "refusal"
    assert _agent_message_texts(conn) == []
    failure_frames = [
        update
        for _session_id, update in conn.updates
        if getattr(update, "session_update", None)
        in {"tool_call", "tool_call_update"}
    ]
    assert [frame.session_update for frame in failure_frames] == [
        "tool_call",
        "tool_call_update",
    ]
    assert failure_frames[1].tool_call_id == failure_frames[0].tool_call_id
    assert failure_frames[1].status == "failed"
    assert state.is_running is False


@pytest.mark.asyncio
async def test_required_agent_exception_calls_stop_required_for_agent(
    monkeypatch,
):
    """The exception-handler branch must terminalize the owned required
    record before returning, mirroring the success-path sibling a few lines
    above it in server.py. Without this, a parent that raises out of
    run_conversation while a required delegation is pending leaves that
    record (and any still-running child work) resident forever: nothing
    else ever calls stop_required_for_agent for it, since this session gets
    no further turn."""

    class _ExceptionAgent(_FakeAgent):
        _required_delegation_launching = True

        def _has_unconsumed_required_delegations(self):
            return True

        def run_conversation(self, **_kwargs):
            raise RuntimeError("post-dispatch failure")

    fake = _ExceptionAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=_NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = _CaptureConn()
    acp_agent.on_connect(conn)
    monkeypatch.setattr(
        acp_agent, "_ensure_delegation_watcher", lambda _loop: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.running_for_session",
        lambda session_key, since_ts=None: [],
    )

    stop_calls = []

    def _record_stop(agent, reason="parent stopped"):
        stop_calls.append((agent, reason))
        return 0

    monkeypatch.setattr(
        "tools.async_delegation.stop_required_for_agent", _record_stop
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="required work")],
    )

    assert response.stop_reason == "refusal"
    # A later, separate safety net further downstream in `prompt()` also
    # terminalizes on the returned error code, so this may legitimately fire
    # more than once (stop_required_for_agent is idempotent past the first
    # terminalization). What must hold is that the except-branch boundary
    # itself calls it immediately, with its own distinct reason, instead of
    # silently relying on that later net ever being reached.
    assert stop_calls, "stop_required_for_agent was never invoked"
    assert all(agent is fake for agent, _reason in stop_calls)
    reasons = [reason for _agent, reason in stop_calls]
    assert any(
        "required child observation completed" in reason for reason in reasons
    )


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
