"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import json
import warnings
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate

from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    _send_update,
    flush_open_tool_calls,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None



    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids

    def test_completion_events_pair_same_name_calls_by_id_exactly_once(
        self, mock_conn, event_loop_fixture
    ):
        tool_call_ids = {}
        tool_call_meta = {}
        progress_cb = make_tool_progress_cb(
            mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update") as send_update:
            progress_cb(
                "tool.started", "terminal", "$ first", {"command": "first"},
                tool_call_id="call-first",
            )
            progress_cb(
                "tool.started", "terminal", "$ second", {"command": "second"},
                tool_call_id="call-second",
            )
            progress_cb(
                "tool.completed", "terminal", None, None,
                tool_call_id="call-second", result='{"output":"second"}',
            )
            progress_cb(
                "tool.completed", "terminal", None, None,
                tool_call_id="call-first", result='{"output":"first"}',
            )

            # The next-step replay and end-turn safety flush must not duplicate
            # terminal updates already projected by tool.completed.
            step_cb(1, [
                {"name": "terminal", "result": '{"output":"first"}'},
                {"name": "terminal", "result": '{"output":"second"}'},
            ])
            assert flush_open_tool_calls(
                mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
            ) == 0

        updates = [call.args[3] for call in send_update.call_args_list]
        assert [update.session_update for update in updates] == [
            "tool_call", "tool_call", "tool_call_update", "tool_call_update",
        ]
        assert [update.tool_call_id for update in updates] == [
            "call-first", "call-second", "call-second", "call-first",
        ]
        assert [update.status for update in updates[2:]] == ["completed", "completed"]

    def test_completion_render_failure_stays_tracked_for_end_turn_flush(
        self, mock_conn, event_loop_fixture
    ):
        tool_call_ids = {}
        tool_call_meta = {}
        progress_cb = make_tool_progress_cb(
            mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update") as send_update:
            progress_cb(
                "tool.started", "terminal", "$ pwd", {"command": "pwd"},
                tool_call_id="call-render-failure",
            )
            with patch(
                "acp_adapter.events.build_tool_complete",
                side_effect=ValueError("bad completion payload"),
            ):
                progress_cb(
                    "tool.completed", "terminal", None, None,
                    tool_call_id="call-render-failure", result="bad result",
                )

            assert list(tool_call_ids["terminal"]) == ["call-render-failure"]
            assert "call-render-failure" in tool_call_meta
            assert flush_open_tool_calls(
                mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
            ) == 1

        updates = [call.args[3] for call in send_update.call_args_list]
        assert [update.session_update for update in updates] == [
            "tool_call", "tool_call_update",
        ]
        assert updates[-1].tool_call_id == "call-render-failure"
        assert updates[-1].status == "completed"

    def test_completion_schedule_failure_stays_tracked_for_end_turn_flush(
        self, mock_conn, event_loop_fixture
    ):
        tool_call_ids = {}
        tool_call_meta = {}
        progress_cb = make_tool_progress_cb(
            mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update"):
            progress_cb(
                "tool.started", "terminal", "$ pwd", {"command": "pwd"},
                tool_call_id="call-schedule-failure",
            )
        with patch("acp_adapter.events._send_update", return_value=False):
            progress_cb(
                "tool.completed", "terminal", None, None,
                tool_call_id="call-schedule-failure", result="done",
            )

        assert list(tool_call_ids["terminal"]) == ["call-schedule-failure"]
        assert "call-schedule-failure" in tool_call_meta


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()



    def test_result_passed_to_build_tool_complete(self, mock_conn, event_loop_fixture):
        """Tool result from prev_tools dict is forwarded to build_tool_complete."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-xyz789"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Provide a result string in the tool info dict
            cb(1, [{"name": "terminal", "result": '{"output": "hello"}'}])

        mock_btc.assert_called_once_with(
            "tc-xyz789", "terminal", result='{"output": "hello"}', function_args=None, snapshot=None
        )

    def test_native_vision_list_result_emits_completion_location(self, mock_conn, event_loop_fixture, tmp_path):
        from collections import deque
        from agent.turn_iteration_prep import _previous_tool_round

        image_path = tmp_path / "native.png"
        tool_call_ids = {"vision_analyze": deque(["tc-native-vision"])}
        tool_call_meta = {"tc-native-vision": {"args": {"image_url": str(image_path)}}}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, tool_call_meta)
        native_content = [
            {"type": "text", "text": "Image loaded into your context."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]

        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-native-vision",
                    "function": {
                        "name": "vision_analyze",
                        "arguments": json.dumps({"image_url": str(image_path), "question": "describe"}),
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call-native-vision", "content": native_content},
        ]

        prev_tools = _previous_tool_round(messages)
        assert isinstance(prev_tools[0]["arguments"], str)

        with patch("acp_adapter.events._send_update") as send_update:
            cb(1, prev_tools)

        update = send_update.call_args.args[3]
        assert update.locations[0].path == str(image_path)



    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot"):
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_send.assert_called_once()

    def test_todo_completion_emits_native_plan_update_after_tool_completion(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"todo": deque(["tc-todo"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"},'
            '{"id":"patch","content":"Patch renderer","status":"in_progress"},'
            '{"id":"old","content":"Drop stale task","status":"cancelled"}'
            '],"summary":{"total":3}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": "todo", "result": todo_result}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call_update",
            "plan",
        ]
        plan = updates[1]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == [
            "Inspect ACP",
            "Patch renderer",
            "[cancelled] Drop stale task",
        ]
        assert [entry.status for entry in plan.entries] == ["completed", "in_progress", "completed"]
        assert [entry.priority for entry in plan.entries] == ["medium", "medium", "medium"]




# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

class TestSendUpdate:
    def test_scheduler_failure_closes_update_coroutine(self, event_loop_fixture):
        """If run_coroutine_threadsafe raises, _send_update must close the coro."""
        created = {"coro": None}

        async def _session_update(session_id, update):
            return None

        conn = MagicMock()

        def _capture_update(session_id, update):
            created["coro"] = _session_update(session_id, update)
            return created["coro"]

        conn.session_update = _capture_update

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                _send_update(conn, "session-1", event_loop_fixture, {"type": "noop"})
            gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        # Only count warnings about THIS test's coroutine; other tests
        #  may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []
