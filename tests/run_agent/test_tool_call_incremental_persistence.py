"""Behavior contracts for incremental tool-call persistence (#49045).

A destructive or process-terminating tool that runs during tool execution
must not lose the just-executed assistant(tool_calls) block or the tool
results that were produced before it fired.  These tests pin the contract:

    1. run_conversation flushes the assistant tool-call turn to the session
       DB BEFORE handing control to _execute_tool_calls (so a tool that
       restarts/kills the process never orphans the tool-call block).
    2. The SEQUENTIAL tool path flushes each tool result to the session DB
       immediately after appending it — BEFORE the next tool dispatches.
    3. The CONCURRENT tool path flushes each tool result in append order.

These exercise the REAL production dispatch surfaces:

    * sequential -> ``run_agent.handle_function_call`` (tool_executor ~1256/1298)
    * concurrent -> ``agent._invoke_tool`` (tool_executor ~539)

Mocking the genuine dispatch surface keeps the tests deterministic (no real
``web_search`` / network) AND mutation-survivable: the ordering assertions
read snapshots captured at flush time, so removing any production flush call
makes the corresponding assertion fail.
"""

import copy
import json
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.agent_runtime_helpers import sanitize_api_messages
from agent.tool_executor import execute_tool_calls_segmented
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.chat_completion_helpers import interruptible_streaming_api_call


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _attach_real_session_db(agent, db_path: Path, session_id: str) -> SessionDB:
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=session_id, source="tui", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    return db


def _durable_messages(db_path: Path, session_id: str) -> list[dict]:
    restarted_db = SessionDB(db_path=db_path)
    try:
        return restarted_db.get_messages_as_conversation(session_id)
    finally:
        restarted_db.close()


def _durable_roles(db_path: Path, session_id: str) -> list[str]:
    return [message["role"] for message in _durable_messages(db_path, session_id)]


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _stream_chunk(*, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_content_after_first_tool_delta_uses_provisional_gate():
    agent = _make_agent()
    delivered = []
    agent.platform = "acp"
    agent.stream_delta_callback = delivered.append
    agent._stream_callback = None
    agent._acp_provisional_stream_active = True
    agent._acp_provisional_stream_buffer = []
    tool_delta = SimpleNamespace(
        index=0,
        id="delegate-1",
        type="function",
        function=SimpleNamespace(name="delegate_task", arguments="{}"),
        extra_content=None,
    )
    agent.client.chat.completions.create.return_value = iter([
        _stream_chunk(tool_calls=[tool_delta]),
        _stream_chunk(content="candidate after tool"),
        _stream_chunk(finish_reason="tool_calls"),
    ])

    response = interruptible_streaming_api_call(
        agent, {"model": "test/model", "messages": []}
    )

    assert response.choices[0].message.tool_calls[0].function.name == "delegate_task"
    assert delivered == []
    assert agent._acp_provisional_stream_buffer == [
        ("stream_delta", "candidate after tool")
    ]


# ---------------------------------------------------------------------------
# Contract 1: run_conversation persists the assistant tool-call block BEFORE
# tool execution begins.
# ---------------------------------------------------------------------------
def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    # Record a deep snapshot of the message list at every flush so the
    # assertion does not depend on later mutations.
    flush_snapshots: list[list] = []

    def _record_flush(messages, conversation_history=None):
        flush_snapshots.append(copy.deepcopy(messages))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    # Capture observations at execute time into module-level lists rather than
    # asserting inside _execute_tool_calls — run_conversation's outer loop
    # swallows exceptions, so an in-callback assertion would never surface.
    executed = {"count": 0}
    snapshot_at_execute: list = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed["count"] += 1
        # Record the DB state observed at the moment tool execution begins.
        snapshot_at_execute.append(
            copy.deepcopy(flush_snapshots[-1]) if flush_snapshots else None
        )
        # Simulate the tool producing a result (as the real path would).
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert executed["count"] == 1, "_execute_tool_calls was never reached"
    # The assistant tool-call block MUST have been flushed before execution.
    last = snapshot_at_execute[0]
    assert last is not None, "no flush occurred before tool execution"
    assert last[-1]["role"] == "assistant"
    assert last[-1]["tool_calls"][0]["id"] == "c1"
    assert result["final_response"] == "done"


def test_interim_assistant_is_durable_before_ui_projection_on_abnormal_exit(tmp_path):
    """A visible interim assistant row must survive an immediate process exit.

    ``GeneratorExit`` models an uncatchable turn interruption at the UI bridge:
    no turn finalizer or graceful shutdown persistence is allowed to rescue the
    row after the callback observes it.
    """
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "interim-abnormal-exit"
    db = _attach_real_session_db(agent, db_path, session_id)
    tool_call = _mock_tool_call(call_id="visible-call")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )

    roles_seen_by_ui: list[str] = []

    def _ui_projection(_text, *, already_streamed=False):
        roles_seen_by_ui.extend(_durable_roles(db_path, session_id))
        raise GeneratorExit("simulated process termination after UI projection")

    agent.interim_assistant_callback = _ui_projection
    try:
        with pytest.raises(GeneratorExit, match="simulated process termination"):
            agent.run_conversation("inspect the repository")
    finally:
        db.close()

    assert roles_seen_by_ui == ["user", "assistant"]
    durable = _durable_messages(db_path, session_id)
    assert [message["role"] for message in durable] == ["user", "assistant"]
    assert durable[1]["content"] == "I'll inspect the repository now."
    assert durable[1]["tool_calls"][0]["id"] == "visible-call"

    # Cold-resume reconciliation closes the interrupted call in the provider
    # payload without mutating or duplicating the canonical transcript.
    resumed = sanitize_api_messages(durable)
    assert [message["role"] for message in resumed] == [
        "user",
        "assistant",
        "tool",
    ]
    assert resumed[2]["tool_call_id"] == "visible-call"
    assert len(_durable_messages(db_path, session_id)) == 2


def test_failed_assistant_persist_blocks_ui_projection_and_tool_side_effects():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.interim_assistant_callback = MagicMock()
    agent._execute_tool_calls = MagicMock()

    result = agent.run_conversation("inspect the repository")

    agent.interim_assistant_callback.assert_not_called()
    agent._execute_tool_calls.assert_not_called()
    assert agent.client is not None
    assert agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "session_persistence_failed"
    # No exception was visible (flush returned False), so the cause is
    # unknown — but the machine-readable contract fields must still be set.
    assert result["failure_reason"] == "session_persistence_failed:unknown"
    assert isinstance(result.get("error"), str) and result["error"].strip() != ""


def test_locked_flush_exception_surfaces_locked_cause_in_result_contract():
    """SQLite write-lock contention must surface as a 'locked' cause.

    Gateway contract: result['failure_reason'] is exactly
    'session_persistence_failed:locked' and result['error'] is a non-empty
    string whose wording talks about busy storage, NOT disk space.
    """
    import sqlite3

    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(
        side_effect=sqlite3.OperationalError("database is locked")
    )
    agent.interim_assistant_callback = MagicMock()
    agent._execute_tool_calls = MagicMock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("inspect the repository")

    agent.interim_assistant_callback.assert_not_called()
    agent._execute_tool_calls.assert_not_called()
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "session_persistence_failed"
    assert result["failure_reason"] == "session_persistence_failed:locked"
    assert isinstance(result.get("error"), str) and result["error"].strip() != ""
    assert "busy" in result["error"].lower()
    assert "disk" not in result["error"].lower()


def test_persistence_cause_resets_between_turns():
    """A locked failure on turn 1 must not leak its cause into turn 2."""
    import sqlite3

    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(
        side_effect=sqlite3.OperationalError("database is locked")
    )
    agent._execute_tool_calls = MagicMock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("inspect the repository")
        assert first["failure_reason"] == "session_persistence_failed:locked"

        # Storage recovered but the flush function now reports a bare False
        # (no exception): the stale 'locked' cause must not be reused.
        agent.client.chat.completions.create.side_effect = None
        agent.client.chat.completions.create.return_value = _mock_response(
            content="I'll inspect the repository now.",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="must-not-run-2")],
        )
        agent._flush_messages_to_session_db = MagicMock(return_value=False)
        second = agent.run_conversation("inspect the repository again")

    assert second["turn_exit_reason"] == "session_persistence_failed"
    assert second["failure_reason"] == "session_persistence_failed:unknown"
def test_acp_repaired_delegate_persists_one_observed_result_before_synthesis(
    tmp_path,
    monkeypatch,
):
    from tools import async_delegation as ad
    import tools.delegate_tool as delegate_module
    from hermes_state import SessionDB

    ad._reset_for_tests()
    agent = _make_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    agent._session_db = db
    agent._session_db_created = False
    agent.platform = "acp"
    agent.valid_tool_names = {
        "delegate_task", "delegation_status",
        "delegation_wait", "delegation_cancel",
    }
    agent.tools = _make_tool_defs(*sorted(agent.valid_tool_names))
    delivered = []
    delivered_interim = []
    agent.stream_delta_callback = delivered.append
    agent.interim_assistant_callback = (
        lambda text, **_kwargs: delivered_interim.append(text)
    )
    alias_delta = SimpleNamespace(
        index=0,
        id="delegate-call",
        type="function",
        function=SimpleNamespace(
            name="delegate",
            arguments='{"goal":"child"}',
        ),
        extra_content=None,
    )
    api_calls = {"count": 0}
    child_release = threading.Event()

    def _response(**_kwargs):
        api_calls["count"] += 1
        if api_calls["count"] == 1:
            agent._fire_streamed_codex_commentary(
                "candidate codex commentary"
            )
            return iter([
                _stream_chunk(content="candidate before delegate "),
                _stream_chunk(tool_calls=[alias_delta]),
                _stream_chunk(content="candidate after tool delta"),
                _stream_chunk(finish_reason="tool_calls"),
            ])
        if api_calls["count"] == 2:
            # The parent is allowed iterative supervision while the child
            # runs, but this no-tool candidate is provisional. Release the
            # child only after the candidate is generated; the hard boundary
            # must discard it, observe durably, then make a new synthesis call.
            child_release.set()
            return iter([
                _stream_chunk(content="premature final without child"),
                _stream_chunk(finish_reason="stop"),
            ])
        durable = db.get_messages_as_conversation(agent.session_id)
        assert any(
            msg.get("role") == "tool"
            and msg.get("tool_name") == "delegation_wait"
            for msg in durable
        ), "required result was not durable before the synthesis call"
        return iter([
            _stream_chunk(content="final from child"),
            _stream_chunk(finish_reason="stop"),
        ])

    agent.client.chat.completions.create.side_effect = _response
    agent._repair_tool_call = MagicMock(
        side_effect=lambda name: "delegate_task" if name == "delegate" else None
    )
    child = SimpleNamespace(
        _subagent_id="child-real-execute",
        _delegate_role="leaf",
        interrupt=MagicMock(),
    )

    def _build_child_agent(*_args, **_kwargs):
        with agent._active_children_lock:
            agent._active_children.append(child)
        return child

    monkeypatch.setattr(
        delegate_module, "_build_child_agent", _build_child_agent
    )
    monkeypatch.setattr(
        delegate_module,
        "_run_single_child",
        lambda task_index, *_args, **_kwargs: (
            child_release.wait(timeout=2)
            and {
                "task_index": task_index,
                "status": "completed",
                "summary": "child evidence",
                "api_calls": 1,
                "duration_seconds": 0,
            }
        ),
    )
    monkeypatch.setattr(
        delegate_module,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("delegate this")
    finally:
        child_release.set()
        ad._reset_for_tests()

    assert result["final_response"] == "final from child"
    # Existing callback protocol closes the pre-tool response box with None
    # and prepends one paragraph break to the first post-tool delta.
    assert delivered[0] is None
    delivered_text = "".join(
        item for item in delivered if isinstance(item, str)
    )
    assert delivered_text.strip() == "final from child"
    assert "candidate before delegate" not in delivered_text
    assert "candidate after tool delta" not in delivered_text
    assert "premature final without child" not in delivered_text
    assert delivered_interim == []
    messages = result["messages"]
    wait_assistants = [
        msg for msg in messages
        if msg.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == "delegation_wait"
            for tc in msg.get("tool_calls", [])
        )
    ]
    wait_results = [
        msg for msg in messages
        if msg.get("role") == "tool" and msg.get("name") == "delegation_wait"
    ]
    assert len(wait_assistants) == 1
    assert len(wait_results) == 1
    assert "child evidence" in wait_results[0]["content"]
    assert getattr(agent, "_last_content_with_tools", None) is None
    replay = db.get_messages_as_conversation(agent.session_id)
    replay_wait_assistants = [
        msg for msg in replay
        if msg.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == "delegation_wait"
            for tc in msg.get("tool_calls", [])
        )
    ]
    replay_wait_results = [
        msg for msg in replay
        if (
            msg.get("role") == "tool"
            and msg.get("tool_name") == "delegation_wait"
        )
    ]
    assert len(replay_wait_assistants) == 1
    assert len(replay_wait_results) == 1
    assert "child evidence" in replay_wait_results[0]["content"]
    assert "candidate before delegate" not in json.dumps(replay)
    assert "candidate after tool delta" not in json.dumps(replay)


def test_required_child_can_be_supervised_iteratively_before_final_join(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    import tools.delegate_tool as delegate_module
    import tools.delegation_live_log as live_log

    ad._reset_for_tests()
    agent = _make_agent()
    agent._session_db = SessionDB(db_path=tmp_path / "iterative-state.db")
    agent._session_db_created = False
    agent.platform = "acp"
    agent.valid_tool_names = {
        "delegate_task",
        "delegation_status",
        "delegation_wait",
        "delegation_cancel",
    }
    agent.tools = _make_tool_defs(*sorted(agent.valid_tool_names))
    child_release = threading.Event()
    premature_candidate_returned = threading.Event()
    api_calls = {"count": 0}
    delegation = {}

    def _response(**_kwargs):
        api_calls["count"] += 1
        call_number = api_calls["count"]
        if call_number == 1:
            return _mock_response(
                content="premature launch prose",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="delegate_task",
                        arguments='{"goal":"child"}',
                        call_id="launch",
                    )
                ],
            )
        if call_number in {2, 3}:
            delegation_id = delegation.setdefault(
                "id",
                ad.list_unconsumed_required(agent)[0]["delegation_id"],
            )
        if call_number == 2:
            return _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="delegation_status",
                        arguments=json.dumps(
                            {"delegation_id": delegation_id}
                        ),
                        call_id="status",
                    )
                ],
            )
        if call_number == 3:
            return _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="delegation_wait",
                        arguments=json.dumps(
                            {
                                "delegation_id": delegation_id,
                                "timeout_seconds": 0,
                            }
                        ),
                        call_id="model-wait",
                    )
                ],
            )
        if call_number == 4:
            premature_candidate_returned.set()
            return _mock_response(
                content="premature final without child evidence",
                finish_reason="stop",
            )
        return _mock_response(
            content="final synthesized child evidence",
            finish_reason="stop",
        )

    agent.client.chat.completions.create.side_effect = _response
    child = SimpleNamespace(
        _subagent_id="child-iterative",
        _delegate_role="leaf",
        interrupt=MagicMock(),
        close=MagicMock(),
        tool_progress_callback=None,
    )

    def _build_child(*_args, **_kwargs):
        with agent._active_children_lock:
            agent._active_children.append(child)
        return child

    def _run_child(task_index, *_args, **_kwargs):
        child_release.wait(timeout=10)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "child evidence",
            "api_calls": 1,
            "duration_seconds": 0,
        }

    monkeypatch.setattr(delegate_module, "_build_child_agent", _build_child)
    monkeypatch.setattr(delegate_module, "_run_single_child", _run_child)
    monkeypatch.setattr(
        live_log,
        "create_live_transcripts",
        lambda *_args, **_kwargs: (None, [], []),
    )
    monkeypatch.setattr(
        delegate_module,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        },
    )

    result_holder = {}

    def _run_parent():
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result_holder["result"] = agent.run_conversation(
                "delegate and supervise"
            )

    parent_thread = threading.Thread(target=_run_parent, daemon=True)
    try:
        parent_thread.start()
        assert premature_candidate_returned.wait(timeout=3)
        # The no-tool candidate cannot end the turn while the child is live.
        parent_thread.join(timeout=0.1)
        assert parent_thread.is_alive()
        child_release.set()
        parent_thread.join(timeout=5)
        assert not parent_thread.is_alive()
    finally:
        child_release.set()
        ad._reset_for_tests()

    result = result_holder["result"]
    assert result["final_response"] == "final synthesized child evidence"
    assert api_calls["count"] == 5
    tool_names = [
        message.get("name")
        for message in result["messages"]
        if message.get("role") == "tool"
    ]
    assert "delegation_status" in tool_names
    assert "delegation_wait" in tool_names
    assert tool_names.count("delegation_wait") == 2
    control_results = [
        json.loads(message["content"])
        for message in result["messages"]
        if message.get("role") == "tool"
        and message.get("name") in {"delegation_status", "delegation_wait"}
    ]
    assert control_results
    assert all(payload.get("status") != "unavailable" for payload in control_results)
    replay = json.dumps(result["messages"])
    assert "child evidence" in replay
    assert "premature launch prose" not in replay
    assert "premature final without child evidence" not in replay


def test_required_observation_atomic_second_row_failure_rolls_back_and_retries_once(
    tmp_path,
):
    from agent.conversation_loop import _observe_required_delegations
    from tools import async_delegation as ad
    from hermes_state import SessionDB

    ad._reset_for_tests()
    agent = _make_agent()
    agent.platform = "acp"
    agent._current_turn_id = "turn-1"
    db = SessionDB(db_path=tmp_path / "atomic-state.db")
    agent._session_db = db
    agent._session_db_created = False
    dispatch = ad.dispatch_async_delegation_batch(
        goals=["child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=lambda: {
            "results": [{"status": "completed", "summary": "child evidence"}],
            "total_duration_seconds": 0,
        },
        required=True,
    )
    delegation_id = dispatch["delegation_id"]
    assert ad.wait_required(
        agent, delegation_id, timeout_seconds=1.0
    )["terminal"]
    messages = []
    real_insert = db._insert_message_rows

    def _fail_after_first_row(conn, session_id, pending):
        real_insert(conn, session_id, pending[:1])
        raise RuntimeError("injected second-row failure")

    try:
        with (
            patch.object(
                db,
                "_insert_message_rows",
                side_effect=_fail_after_first_row,
            ),
            pytest.raises(RuntimeError, match="second-row failure"),
        ):
            _observe_required_delegations(agent, messages, [])
        status = ad.required_status(agent, delegation_id)
        assert messages == []
        assert status["terminal"] is True
        assert status["consumed"] is False
        failed_replay = db.get_messages_as_conversation(agent.session_id)
        assert not any(
            message.get("tool_calls") for message in failed_replay
        )
        assert not any(
            message.get("tool_call_id") == f"required_wait_{delegation_id}"
            for message in failed_replay
        )

        assert _observe_required_delegations(agent, messages, []) is True
        replay = db.get_messages_as_conversation(agent.session_id)
        consumed = ad.required_status(agent, delegation_id)["consumed"]
    finally:
        ad._reset_for_tests()

    wait_calls = [
        message for message in replay
        if message.get("role") == "assistant"
        and any(
            call.get("id") == f"required_wait_{delegation_id}"
            for call in message.get("tool_calls", [])
        )
    ]
    wait_results = [
        message for message in replay
        if (
            message.get("role") == "tool"
            and message.get("tool_call_id")
            == f"required_wait_{delegation_id}"
        )
    ]
    assert len(wait_calls) == 1
    assert len(wait_results) == 1
    assert consumed is True


def test_model_wait_keeps_gate_closed_until_one_atomic_synthetic_observation(
    tmp_path,
):
    from agent.conversation_loop import _observe_required_delegations
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    from tools.delegate_tool import _required_control

    ad._reset_for_tests()
    agent = _make_agent()
    agent.platform = "acp"
    agent._delegate_depth = 0
    agent._current_turn_id = "turn-model-wait"
    agent._session_db = SessionDB(db_path=tmp_path / "model-wait-state.db")
    agent._session_db_created = False
    dispatch = ad.dispatch_async_delegation_batch(
        goals=["child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=lambda: {
            "results": [{
                "status": "completed",
                "summary": "canonical child evidence",
            }],
            "total_duration_seconds": 0,
        },
        required=True,
    )
    delegation_id = dispatch["delegation_id"]
    assert ad.wait_required(
        agent, delegation_id, timeout_seconds=1.0
    )["terminal"]
    wait_call = _mock_tool_call(
        name="delegation_wait",
        arguments=json.dumps({
            "delegation_id": delegation_id,
            "timeout_seconds": 0,
        }),
        call_id="model-wait",
    )
    wait_content = _required_control(
        "wait",
        {"delegation_id": delegation_id, "timeout_seconds": 0},
        agent,
    )
    wait_payload = json.loads(wait_content)
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": wait_call.id,
            "type": "function",
            "function": {
                "name": wait_call.function.name,
                "arguments": wait_call.function.arguments,
            },
        }],
    }, make_tool_result_message(
        "delegation_wait", wait_content, wait_call.id
    )]

    try:
        assert wait_payload["terminal"] is True
        assert wait_payload["observation_pending"] is True
        assert "result" not in wait_payload
        assert ad.required_status(agent, delegation_id)["consumed"] is False
        assert _observe_required_delegations(agent, messages, []) is True
        replay = agent._session_db.get_messages_as_conversation(
            agent.session_id
        )
        consumed = ad.required_status(agent, delegation_id)["consumed"]
    finally:
        ad._reset_for_tests()

    evidence_results = [
        message for message in replay
        if (
            message.get("role") == "tool"
            and "canonical child evidence" in str(message.get("content"))
        )
    ]
    assert len(evidence_results) == 1
    assert evidence_results[0]["tool_call_id"] == (
        f"required_wait_{delegation_id}"
    )
    assert consumed is True


def test_model_cancel_keeps_result_for_one_atomic_synthetic_observation(
    tmp_path,
):
    from agent.conversation_loop import _observe_required_delegations
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    from tools.delegate_tool import _required_control

    ad._reset_for_tests()
    release = threading.Event()
    agent = _make_agent()
    agent.platform = "acp"
    agent._delegate_depth = 0
    agent._current_turn_id = "turn-model-cancel"
    agent._session_db = SessionDB(
        db_path=tmp_path / "model-cancel-state.db"
    )
    agent._session_db_created = False
    dispatch = ad.dispatch_async_delegation_batch(
        goals=["child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=lambda: (
            release.wait(timeout=2)
            and {
                "results": [{
                    "task_index": 0,
                    "child_id": "child-cancel",
                    "status": "completed",
                    "summary": "late result",
                }],
                "total_duration_seconds": 0,
            }
        ),
        interrupt_fn=release.set,
        child_ids=["child-cancel"],
        required=True,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="child-cancel",
        current_tool=None,
        activity="started",
        meaningful=False,
        state="running",
    )
    cancel_content = _required_control(
        "cancel", {"delegation_id": delegation_id}, agent
    )
    cancel_payload = json.loads(cancel_content)
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "model-cancel",
            "type": "function",
            "function": {
                "name": "delegation_cancel",
                "arguments": json.dumps({
                    "delegation_id": delegation_id
                }),
            },
        }],
    }, make_tool_result_message(
        "delegation_cancel", cancel_content, "model-cancel"
    )]

    try:
        assert cancel_payload["terminal"] is True
        assert cancel_payload["observation_pending"] is True
        assert "result" not in cancel_payload
        assert _observe_required_delegations(agent, messages, []) is True
        replay = agent._session_db.get_messages_as_conversation(
            agent.session_id
        )
    finally:
        release.set()
        ad._reset_for_tests()

    evidence_results = [
        message for message in replay
        if (
            message.get("role") == "tool"
            and message.get("tool_name") == "delegation_wait"
            and '"status": "cancelled"' in str(message.get("content"))
        )
    ]
    assert len(evidence_results) == 1


def test_required_join_emits_acp_visible_wait_activity_for_queued_child():
    from agent.conversation_loop import _observe_required_delegations
    from tools import async_delegation as ad

    ad._reset_for_tests()
    release = threading.Event()
    agent = _make_agent()
    agent.platform = "acp"
    agent._current_turn_id = "turn-visible-wait"
    visible_waits = []
    visible_wait_seen = threading.Event()

    def _record_visible_wait(text):
        visible_waits.append(text)
        visible_wait_seen.set()

    agent._emit_wait_notice = _record_visible_wait
    agent._persist_required_observation_pair = MagicMock()
    dispatch = ad.dispatch_async_delegation_batch(
        goals=["queued child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=lambda: (
            release.wait(timeout=2)
            and {
                "results": [],
                "total_duration_seconds": 0,
            }
        ),
        interrupt_fn=release.set,
        child_ids=["queued-child"],
        required=True,
        no_progress_timeout_seconds=1000,
        start_timeout_seconds=5,
    )
    messages = []
    observation = {}
    observer = threading.Thread(
        target=lambda: observation.setdefault(
            "ok", _observe_required_delegations(
                agent, messages, [], wait_for_pending=True
            )
        ),
        daemon=True,
    )
    observer.start()
    try:
        assert visible_wait_seen.wait(timeout=2), "required wait was not surfaced"
        release.set()
        observer.join(timeout=2)
        assert not observer.is_alive(), "required observation did not finish"
        terminal = ad.required_status(
            agent, dispatch["delegation_id"]
        )
    finally:
        release.set()
        observer.join(timeout=2)
        ad._reset_for_tests()

    assert observation["ok"] is True
    assert visible_waits
    assert dispatch["delegation_id"] in visible_waits[0]
    assert "queued" in visible_waits[0]
    assert terminal["terminal"] is True


def test_terminal_wrapper_hard_joins_and_replaces_stale_rollback_messages():
    from agent.conversation_loop import _required_safe_terminal_result
    from tools import async_delegation as ad

    ad._reset_for_tests()
    release = threading.Event()
    agent = _make_agent()
    agent.platform = "acp"
    agent._current_turn_id = "turn-terminal-wrapper"
    agent._persist_required_observation_pair = MagicMock()
    dispatch = ad.dispatch_async_delegation_batch(
        goals=["child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=lambda: (
            release.wait(timeout=2)
            and {
                "results": [{
                    "status": "completed",
                    "summary": "joined evidence",
                }],
                "total_duration_seconds": 0,
            }
        ),
        required=True,
    )
    live_messages = [{"role": "user", "content": "live"}]
    stale_messages = [{"role": "user", "content": "rolled back"}]
    result_holder = {}

    def _finish():
        result_holder["result"] = _required_safe_terminal_result(
            agent,
            {
                "final_response": "explicit failure",
                "messages": stale_messages,
                "api_calls": 2,
                "completed": False,
                "failed": True,
                "error": "retry_exhausted",
            },
            live_messages,
            [],
        )

    thread = threading.Thread(target=_finish)
    thread.start()
    threading.Event().wait(0.05)
    assert thread.is_alive()
    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    try:
        terminal = result_holder["result"]
        assert terminal["final_response"] == "explicit failure"
        assert terminal["messages"] is live_messages
        assert any(
            message.get("name") == "delegation_wait"
            and "joined evidence" in str(message.get("content"))
            for message in live_messages
        )
        assert ad.required_status(
            agent, dispatch["delegation_id"]
        )["consumed"] is True
    finally:
        ad._reset_for_tests()


def test_mixed_guardrail_after_required_dispatch_fails_closed_without_prose():
    from tools import async_delegation as ad

    ad._reset_for_tests()
    agent = _make_agent()
    agent.platform = "acp"
    agent.valid_tool_names = {"delegate_task", "web_search"}
    agent.tools = _make_tool_defs("delegate_task", "web_search")
    agent._persist_required_observation_pair = MagicMock()
    calls = [
        _mock_tool_call(
            name="delegate_task",
            arguments='{"goal":"child"}',
            call_id="delegate-mixed",
        ),
        _mock_tool_call(
            name="web_search",
            arguments='{"query":"second tool"}',
            call_id="guarded-second",
        ),
    ]
    agent.client.chat.completions.create.return_value = _mock_response(
        content="candidate must not escape",
        finish_reason="tool_calls",
        tool_calls=calls,
    )

    def _execute(_assistant_message, messages, *_args):
        dispatch = ad.dispatch_async_delegation_batch(
            goals=["child"],
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key=agent.session_id,
            parent_session_id=agent.session_id,
            parent_turn_id=agent._current_turn_id,
            runner=lambda: {
                "results": [{
                    "status": "completed",
                    "summary": "immediate child evidence",
                }],
                "total_duration_seconds": 0,
            },
            required=True,
        )
        messages.append(make_tool_result_message(
            "delegate_task",
            json.dumps({
                "status": "dispatched",
                "delegation_id": dispatch["delegation_id"],
            }),
            "delegate-mixed",
        ))
        messages.append(make_tool_result_message(
            "web_search",
            json.dumps({"error": "guarded"}),
            "guarded-second",
        ))
        agent._tool_guardrail_halt_decision = SimpleNamespace(
            tool_name="web_search",
            code="test_guardrail",
        )

    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(
                agent, "_execute_tool_calls", side_effect=_execute
            ),
        ):
            result = agent.run_conversation("delegate and guarded tool")
    finally:
        ad._reset_for_tests()

    assert result["final_response"] is None
    assert result["error"] == (
        "tool_guardrail_halt_after_required_delegation"
    )
    assert "candidate must not escape" not in json.dumps(
        result["messages"]
    )
    assert "immediate child evidence" in json.dumps(result["messages"])


def test_required_dispatch_processing_error_observes_child_and_clears_latch(
    tmp_path,
):
    from tools import async_delegation as ad
    from hermes_state import SessionDB

    ad._reset_for_tests()
    agent = _make_agent()
    agent._session_db = SessionDB(db_path=tmp_path / "error-state.db")
    agent._session_db_created = False
    agent.platform = "acp"
    # The mocked exception originates in this test module rather than one of
    # the production-local processing modules used by phase classification.
    # Bound the loop so this test isolates one recovery/observation cycle.
    agent.max_iterations = 2
    agent.valid_tool_names = {"delegate_task"}
    agent.tools = _make_tool_defs("delegate_task")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="candidate",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(
            name="delegate_task", arguments="{}", call_id="delegate-error"
        )],
    )

    def _execute(_assistant_message, messages, *_args):
        dispatch = ad.dispatch_async_delegation_batch(
            goals=["child"],
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key=agent.session_id,
            parent_session_id=agent.session_id,
            parent_turn_id=agent._current_turn_id,
            runner=lambda: {
                "results": [{
                    "status": "completed",
                    "summary": "evidence before local failure",
                }],
                "total_duration_seconds": 0,
            },
            required=True,
        )
        messages.append(make_tool_result_message(
            "delegate_task",
            json.dumps({
                "status": "dispatched",
                "mode": "required",
                "delegation_id": dispatch["delegation_id"],
            }),
            "delegate-error",
        ))
        raise RuntimeError("local post-dispatch failure")

    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        ):
            result = agent.run_conversation("delegate then fail locally")
    finally:
        ad._reset_for_tests()

    assert result["completed"] is True
    assert "local post-dispatch failure" in result["final_response"]
    wait_results = [
        message for message in result["messages"]
        if (
            message.get("role") == "tool"
            and message.get("name") == "delegation_wait"
        )
    ]
    assert len(wait_results) == 1
    assert "evidence before local failure" in wait_results[0]["content"]
    assert agent._required_delegation_launching is False
    assert agent._acp_provisional_stream_active is False


def test_required_launch_latch_clears_after_truncated_validation_exit():
    agent = _make_agent()
    agent.platform = "acp"
    agent.valid_tool_names = {"delegate_task"}
    agent.tools = _make_tool_defs("delegate_task")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="candidate",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(
            name="delegate_task",
            arguments='{"goal":"unfinished',
            call_id="bad-delegate",
        )],
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        failed = agent.run_conversation("delegate malformed")

    assert failed["partial"] is True
    assert agent._required_delegation_launching is False
    assert agent._acp_provisional_stream_active is False

    agent.client.chat.completions.create.return_value = _mock_response(
        content="clean next turn", finish_reason="stop"
    )
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        recovered = agent.run_conversation("new turn")
    assert recovered["final_response"] == "clean next turn"


def test_next_acp_turn_cancels_and_consumes_orphaned_required_owner():
    from tools import async_delegation as ad

    ad._reset_for_tests()
    agent = _make_agent()
    agent.platform = "acp"
    agent._current_turn_id = "old-turn"
    started = threading.Event()
    release = threading.Event()
    interrupted = []

    def _runner():
        started.set()
        release.wait(timeout=2)
        return {
            "results": [{"status": "completed", "summary": "late child"}],
            "total_duration_seconds": 0,
        }

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["orphaned child"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_turn_id=agent._current_turn_id,
        runner=_runner,
        interrupt_fn=lambda: (interrupted.append("stop"), release.set()),
        required=True,
    )
    delegation_id = dispatch["delegation_id"]
    assert started.wait(timeout=1)
    agent.client.chat.completions.create.return_value = _mock_response(
        content="clean next turn", finish_reason="stop"
    )

    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("new turn")
        with ad._records_lock:
            record = dict(ad._records[delegation_id])
    finally:
        release.set()
        ad._reset_for_tests()

    assert result["final_response"] == "clean next turn"
    assert agent._current_turn_id != "old-turn"
    assert record["status"] == "cancelled"
    assert record["consumed_at"] is not None
    assert interrupted == ["stop"]


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through run_agent.handle_function_call
# (the real production surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_sequential_flushes_each_tool_result_before_next_dispatch():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    # Ordered event log interleaving real dispatches and DB flushes.
    events: list = []

    def _fake_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # The result for call N must have been flushed before call N+1 fires.
        events.append(("dispatch", kwargs.get("tool_call_id")))
        return f"result-{kwargs.get('tool_call_id')}"

    def _record_flush(flush_messages, conversation_history=None):
        # Snapshot the tail tool result that triggered this flush.
        tail = flush_messages[-1]
        events.append(("flush", tail.get("role"), tail.get("tool_call_id")))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # The mock proves we exercised the REAL sequential dispatch surface.
    assert disp.call_count == 2, "sequential path did not dispatch via handle_function_call"

    # Both tool results landed, in order.
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Ordering contract: each tool result is flushed AFTER its own dispatch
    # and BEFORE the next dispatch. Expected interleaving:
    #   dispatch c1 -> flush c1 -> dispatch c2 -> flush c2
    assert events == [
        ("dispatch", "c1"),
        ("flush", "tool", "c1"),
        ("dispatch", "c2"),
        ("flush", "tool", "c2"),
    ]


def test_sequential_keyboard_interrupt_emits_results_for_all_calls():
    """A KeyboardInterrupt mid-batch must not leave dangling tool_calls.

    When a tool handler raises KeyboardInterrupt, the sequential executor
    re-raises to abort the turn — but it must first append a tool result for
    the interrupted call AND every remaining call, or the assistant tool-call
    turn is left without matching tool results (a message-role alternation
    violation that malforms the next provider request). Mirrors the
    cooperative-interrupt and concurrent paths, which already do this.
    """
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
        _mock_tool_call(name="web_search", call_id="c3"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    def _interrupt_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # First tool raises a hard interrupt mid-batch.
        raise KeyboardInterrupt()

    agent._flush_messages_to_session_db = MagicMock()

    with (
        patch("run_agent.handle_function_call", side_effect=_interrupt_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # Every call_id has a matching tool result — alternation preserved.
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["c1", "c2", "c3"]
    # The results are marked as cancelled, not fabricated successes.
    assert all("cancelled" in m["content"].lower() for m in tool_results)


@pytest.mark.parametrize("executor_mode", ["sequential", "concurrent"])
def test_tool_result_is_durable_before_ui_completion_on_abnormal_exit(
    tmp_path,
    executor_mode,
):
    """A visible tool completion must already exist in the canonical DB."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = f"tool-result-abnormal-exit-{executor_mode}"
    db = _attach_real_session_db(agent, db_path, session_id)
    tool_call = _mock_tool_call(call_id="visible-call")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": "I'll inspect the repository now.",
            "tool_calls": [
                {
                    "id": "visible-call",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        },
    ]
    agent._flush_messages_to_session_db(messages)

    roles_seen_by_ui: list[str] = []

    def _ui_completion(*_args):
        roles_seen_by_ui.extend(_durable_roles(db_path, session_id))
        raise GeneratorExit("simulated process termination after tool completion")

    agent.tool_complete_callback = _ui_completion
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    dispatch_patch = (
        patch("run_agent.handle_function_call", return_value="repository result")
        if executor_mode == "sequential"
        else patch.object(agent, "_invoke_tool", return_value="repository result")
    )
    try:
        with (
            dispatch_patch,
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
            pytest.raises(GeneratorExit, match="simulated process termination"),
        ):
            if executor_mode == "sequential":
                agent._execute_tool_calls_sequential(
                    assistant_message,
                    messages,
                    "task-1",
                )
            else:
                agent._execute_tool_calls_concurrent(
                    assistant_message,
                    messages,
                    "task-1",
                )
    finally:
        db.close()

    expected_roles = ["user", "assistant", "tool"]
    assert roles_seen_by_ui == expected_roles
    durable = _durable_messages(db_path, session_id)
    assert [message["role"] for message in durable] == expected_roles
    assert durable[2]["tool_call_id"] == "visible-call"
    assert durable[2]["content"] == "repository result"


@pytest.mark.parametrize("executor_mode", ["sequential", "concurrent"])
def test_failed_tool_result_persist_blocks_completion_projection(executor_mode):
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="failed-persist")
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.tool_complete_callback = MagicMock()
    dispatch_patch = (
        patch("run_agent.handle_function_call", return_value="repository result")
        if executor_mode == "sequential"
        else patch.object(agent, "_invoke_tool", return_value="repository result")
    )

    with (
        dispatch_patch,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        if executor_mode == "sequential":
            agent._execute_tool_calls_sequential(
                assistant_message,
                messages,
                "task-1",
            )
        else:
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )

    agent.tool_complete_callback.assert_not_called()
    assert getattr(agent, "_incremental_persistence_failed", False) is True


def test_segmented_batch_stops_before_later_segment_after_persist_failure():
    agent = _make_agent()
    first = _mock_tool_call(call_id="first")
    second = _mock_tool_call(call_id="second")
    assistant_message = SimpleNamespace(tool_calls=[first, second])
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock(return_value=False)

    with (
        patch.object(agent, "_invoke_tool", return_value="first result") as invoke,
        patch("run_agent.handle_function_call", return_value="second result") as dispatch,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute_tool_calls_segmented(
            agent,
            assistant_message,
            messages,
            "task-1",
            segments=[("parallel", [first]), ("sequential", [second])],
        )

    invoke.assert_called_once()
    dispatch.assert_not_called()
    assert getattr(agent, "_incremental_persistence_failed", False) is True


# ---------------------------------------------------------------------------
# Contract 3: the CONCURRENT path flushes each collected tool result in append
# order.  Dispatch goes through agent._invoke_tool (the real concurrent
# surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    invoked_ids: list = []

    def _fake_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        invoked_ids.append(tool_call_id)
        return f"result-{tool_call_id}"

    # Each flush must observe exactly one more tool result than the previous
    # flush, in append order — i.e. the tail tool_call_id sequence is c1, c2.
    flushed_tool_ids: list = []
    flush_lengths: list = []

    def _record_flush(flush_messages, conversation_history=None):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])
        flush_lengths.append(len([m for m in flush_messages if m.get("role") == "tool"]))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    # Proves the real concurrent dispatch surface was exercised.
    assert inv.call_count == 2, "concurrent path did not dispatch via _invoke_tool"
    assert sorted(invoked_ids) == ["c1", "c2"]

    # Results appended in deterministic order.
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Each tool result was flushed exactly once, in append order, with the
    # running tool count growing by one each time (1 then 2).  Removing either
    # production flush call breaks one of these assertions.
    assert flushed_tool_ids == ["c1", "c2"]
    assert flush_lengths == [1, 2]


def test_concurrent_timed_out_tool_completion_does_not_advance_last_meaningful_at():
    """Adversarial review follow-up (MEDIUM): a concurrent tool call that hits
    the batch deadline is left running "detached" (agent/tool_executor.py's
    own comment) — the executor gives up on it, but the worker thread may
    still be genuinely wedged. The late/never-verified completion touch for
    that tool must NOT be meaningful=True: stamping it would refresh a
    required-delegation child's no-progress deadline exactly when
    supervision should be tightening, not resetting, and would let a child
    that keeps hitting per-tool timeouts reset its ceiling for free."""
    from tools import async_delegation as ad

    agent = _make_agent()
    owner_token = "owner-timeout"
    agent._required_delegation_owner_token = owner_token
    agent.session_id = "parent-timeout"
    agent._current_turn_id = "turn-timeout"

    runner_release = threading.Event()

    def _runner():
        runner_release.wait(timeout=10)
        return {"results": [{"task_index": 0, "status": "completed"}]}

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["hung tool work"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_owner_token=owner_token,
        parent_turn_id=agent._current_turn_id,
        runner=_runner,
        child_ids=["child-timeout"],
        required=True,
        max_async_children=3,
        no_progress_timeout_seconds=1000.0,
        in_flight_no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    try:
        agent._required_delegation_id = delegation_id
        agent._subagent_id = "child-timeout"
        ad.note_required_progress(
            delegation_id,
            child_id="child-timeout",
            current_tool=None,
            activity="started",
            meaningful=False,
            state="running",
        )

        tool_calls = [_mock_tool_call(name="web_search", call_id="c1")]
        messages: list = []
        assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

        # Sleeps far longer than the configured batch timeout — the executor
        # gives up and reports it timed out while the worker is still
        # genuinely running in the background (daemon thread; exits on its
        # own after 0.3s, well after this test has already asserted).
        def _hung_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
            threading.Event().wait(0.3)
            return "should never be observed by the executor"

        before = time.time()
        with (
            patch.object(agent, "_invoke_tool", side_effect=_hung_invoke),
            patch("agent.tool_executor._resolve_concurrent_tool_timeout", return_value=0.05),
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
        ):
            agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

        # The executor reported the timeout in the returned tool message.
        assert len(messages) == 1
        assert "timed out" in messages[0]["content"]

        with ad._records_lock:
            child = ad._records[delegation_id]["child_supervision"]["child-timeout"]
            last_meaningful_at = child["last_meaningful_at"]

        # last_meaningful_at must still reflect the batch START touch (fired
        # once, unconditionally, before any worker launched) — NOT a fresh
        # stamp from the abandoned tool's completion touch. The batch took
        # ~50ms+ (the configured timeout) to return, so a bug that stamps
        # meaningful=True on the timed-out completion would push
        # last_meaningful_at well past `before + small epsilon`.
        assert last_meaningful_at <= before + 0.03, (
            "last_meaningful_at advanced past the batch start — the "
            "timed-out tool's completion touch was (incorrectly) meaningful"
        )
    finally:
        runner_release.set()
        ad._reset_for_tests()


def test_concurrent_normal_completion_still_advances_last_meaningful_at():
    """Companion to the timeout test above: a tool the executor DID verify
    complete (success or tool-level error) must still count as real
    progress — the fix must not blanket-suppress meaningful=True for every
    concurrent completion, only for the unverified (r is None) ones."""
    from tools import async_delegation as ad

    agent = _make_agent()
    owner_token = "owner-normal"
    agent._required_delegation_owner_token = owner_token
    agent.session_id = "parent-normal"
    agent._current_turn_id = "turn-normal"

    runner_release = threading.Event()

    def _runner():
        runner_release.wait(timeout=10)
        return {"results": [{"task_index": 0, "status": "completed"}]}

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["normal tool work"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=agent.session_id,
        parent_session_id=agent.session_id,
        parent_owner_token=owner_token,
        parent_turn_id=agent._current_turn_id,
        runner=_runner,
        child_ids=["child-normal"],
        required=True,
        max_async_children=3,
        no_progress_timeout_seconds=1000.0,
        in_flight_no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    try:
        agent._required_delegation_id = delegation_id
        agent._subagent_id = "child-normal"
        ad.note_required_progress(
            delegation_id,
            child_id="child-normal",
            current_tool=None,
            activity="started",
            meaningful=False,
            state="running",
        )
        with ad._records_lock:
            # Force the clock stale so a real advance is unambiguous.
            ad._records[delegation_id]["child_supervision"]["child-normal"][
                "last_meaningful_at"
            ] -= 100.0
            before_stale = ad._records[delegation_id]["child_supervision"][
                "child-normal"
            ]["last_meaningful_at"]

        tool_calls = [_mock_tool_call(name="web_search", call_id="c1")]
        messages: list = []
        assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

        with (
            patch.object(agent, "_invoke_tool", return_value="ok result"),
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
        ):
            agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

        with ad._records_lock:
            last_meaningful_at = ad._records[delegation_id]["child_supervision"][
                "child-normal"
            ]["last_meaningful_at"]
        assert last_meaningful_at > before_stale + 50.0
    finally:
        runner_release.set()
        ad._reset_for_tests()
