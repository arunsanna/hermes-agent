"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature that AIAgent expects
for its callbacks. Internally, the callbacks push ACP session updates
to the client via ``conn.session_update()`` using
``asyncio.run_coroutine_threadsafe()`` (since AIAgent runs in a worker
thread while the event loop lives on the main thread).
"""

import asyncio
import json
import logging
import os
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    _text as _tool_text,
    build_tool_complete,
    build_tool_start,
    make_tool_call_id,
)

logger = logging.getLogger(__name__)


def _subagent_updates_enabled() -> bool:
    """Per-child subagent ACP emission gate (default on).

    Set ``HERMES_ACP_SUBAGENT_UPDATES=0`` to fall back to the legacy
    behaviour (one opaque delegate_task tool call, no per-child frames).
    """
    return os.environ.get("HERMES_ACP_SUBAGENT_UPDATES", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _json_loads_maybe_prefix(value: str) -> Any:
    """Parse a JSON object even when Hermes appended a human hint after it."""
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)
        return data


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task/todo panel. The
    Hermes agent already maintains task state through the ``todo`` tool, so the
    ACP adapter should expose that state natively instead of only as a generic
    tool-call transcript block.
    """
    if not isinstance(result, str) or not result.strip():
        return None

    try:
        data = _json_loads_maybe_prefix(result)
    except Exception:
        return None

    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    todos = data["todos"]
    if not todos:
        return AgentPlanUpdate(session_update="plan", entries=[])

    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
        # ACP plans only support pending/in_progress/completed. Preserve
        # cancelled tasks as terminal entries instead of dropping them and
        # making the client's full-list replacement lose visible context.
        "cancelled": "completed",
    }
    entries: list[PlanEntry] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        status = status_map.get(raw_status, "pending")
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=status))

    return AgentPlanUpdate(session_update="plan", entries=entries)


def _send_update(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    update: Any,
) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update),
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and tracks IDs in a FIFO
    queue per tool name so duplicate/parallel same-name calls still complete
    against the correct ACP tool call.  Other event types (``tool.completed``,
    ``reasoning.available``) are silently ignored.
    """

    # Per-child delegate_task subagent calls (subagent_id -> ACP tool call id)
    # plus the last relayed activity snippet per child, folded into the final
    # completion frame. Children run on worker threads; per-key dict ops are
    # safe under the GIL and each child only touches its own key.
    child_calls: Dict[str, str] = {}
    child_last_activity: Dict[str, str] = {}

    def _handle_subagent_event(
        event_type: str, tool_name: str, preview: str, kwargs: Dict[str, Any]
    ) -> None:
        """Translate relayed ``subagent.*`` events into per-child ACP frames.

        delegate_tool relays child lifecycle fully identity-tagged
        (subagent_id/task_index/goal/...). Emission contract for ACP clients
        (Switchboard closes an item on ANY tool_call_update, so intermediate
        frames must not be sent): one ToolCallStart on ``subagent.start`` and
        one final update on ``subagent.complete``; everything in between only
        refreshes the activity snippet included in the completion.
        """
        if not _subagent_updates_enabled():
            return
        sid = kwargs.get("subagent_id")
        if sid is None:
            return
        sid = str(sid)

        if event_type == "subagent.start":
            if sid in child_calls:
                return
            tc_id = make_tool_call_id()
            child_calls[sid] = tc_id
            goal = str(preview or kwargs.get("goal") or "").strip()
            title = "subagent"
            if goal:
                title += ": " + (goal[:120] + ("…" if len(goal) > 120 else ""))
            raw_arguments: Dict[str, Any] = {"subagentId": sid}
            if goal:
                raw_arguments["goal"] = goal[:400]
            for key, out_key in (
                ("task_index", "taskIndex"),
                ("task_count", "taskCount"),
                ("model", "model"),
                ("depth", "depth"),
            ):
                value = kwargs.get(key)
                if value is not None:
                    raw_arguments[out_key] = value
            update = acp.start_tool_call(
                tc_id,
                title,
                kind="execute",
                raw_input={"tool": "subagent", "arguments": raw_arguments},
            )
            _send_update(conn, session_id, loop, update)
            return

        if event_type == "subagent.complete":
            tc_id = child_calls.pop(sid, None)
            last_activity = child_last_activity.pop(sid, None)
            if tc_id is None:
                return
            status_raw = str(kwargs.get("status") or "completed").strip().lower()
            status = (
                "completed"
                if status_raw in {"completed", "complete", "success", "done", ""}
                else "failed"
            )
            parts = []
            for candidate in (preview, kwargs.get("summary"), last_activity):
                text_value = str(candidate or "").strip()
                if text_value and text_value not in parts:
                    parts.append(text_value)
            update = acp.update_tool_call(
                tc_id,
                kind="execute",
                status=status,
                content=[_tool_text("\n\n".join(parts))] if parts else None,
            )
            _send_update(conn, session_id, loop, update)
            return

        # subagent.tool / subagent.thinking / subagent.text / subagent.progress:
        # keep the freshest snippet for the completion frame.
        snippet = str(preview or tool_name or "").strip()
        if snippet and sid in child_calls:
            child_last_activity[sid] = snippet[:500]

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        if isinstance(event_type, str) and event_type.startswith("subagent."):
            _handle_subagent_event(event_type, name, preview, kwargs)
            return
        # Only emit ACP ToolCallStart for tool.started; ignore other event types
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = make_tool_call_id()
        queue = tool_call_ids.get(name)
        if queue is None:
            queue = deque()
            tool_call_ids[name] = queue
        elif isinstance(queue, str):
            queue = deque([queue])
            tool_call_ids[name] = queue
        queue.append(tc_id)

        snapshot = None
        if name in {"write_file", "patch", "skill_manage"}:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        tool_call_meta[tc_id] = {"args": args, "snapshot": snapshot}

        edit_diff = None
        if name in {"write_file", "patch"} and edit_approval_policy_getter is not None:
            try:
                from acp_adapter.edit_approval import build_edit_proposal, should_auto_approve_edit

                proposal = build_edit_proposal(name, args)
                if proposal is not None:
                    policy, cwd = edit_approval_policy_getter()
                    if should_auto_approve_edit(proposal, policy, cwd):
                        edit_diff = proposal
            except Exception:
                logger.debug("Failed to prepare auto-approved ACP edit diff for %s", name, exc_info=True)

        update = build_tool_start(tc_id, name, args, edit_diff=edit_diff)
        _send_update(conn, session_id, loop, update)

    return _tool_progress


# ------------------------------------------------------------------
# Thinking callback
# ------------------------------------------------------------------

def make_thinking_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""

    def _thinking(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_thought_text(text)
        _send_update(conn, session_id, loop, update)

    return _thinking


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if prev_tools and isinstance(prev_tools, list):
            for tool_info in prev_tools:
                tool_name = None
                result = None
                function_args = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                    function_args = tool_info.get("arguments") or tool_info.get("args")
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()
                    meta = tool_call_meta.pop(tc_id, {})
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=str(result) if result is not None else None,
                        function_args=function_args or meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    if tool_name == "todo":
                        plan_update = _build_plan_update_from_todo_result(result)
                        if plan_update is not None:
                            _send_update(conn, session_id, loop, plan_update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)
                elif tool_name:
                    # No queued start for this completion: the pairing FIFO can
                    # drift on long turns (steering/compression rewrite the
                    # message history prev_tools is rebuilt from). Log instead
                    # of silently dropping so wire-level completion loss is
                    # diagnosable; flush_open_tool_calls() closes the inverse
                    # case (started-but-never-completed) at turn end.
                    logger.debug(
                        "ACP completion for %r has no queued tool_call id; dropping",
                        tool_name,
                    )

    return _step


def flush_open_tool_calls(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> int:
    """Close every tool call that never received a completion this turn.

    The name-keyed FIFO pairing in ``make_step_cb`` loses completions on long
    turns (live evidence 2026-07-19: 61 tool_call starts vs 22 updates in one
    session), leaving clients with stuck in-progress items. Called at the end
    of a prompt turn so the wire always converges; results are unknown by then,
    so frames carry status=completed with no content.
    """
    flushed = 0
    for tool_name in list(tool_call_ids.keys()):
        queue = tool_call_ids.pop(tool_name)
        if isinstance(queue, str):
            queue = deque([queue])
        while queue:
            tc_id = queue.popleft()
            meta = tool_call_meta.pop(tc_id, {})
            update = build_tool_complete(
                tc_id,
                tool_name,
                result=None,
                function_args=meta.get("args"),
                snapshot=meta.get("snapshot"),
            )
            _send_update(conn, session_id, loop, update)
            flushed += 1
    if flushed:
        logger.debug("Flushed %d unclosed ACP tool call(s) at turn end", flushed)
    return flushed


# ------------------------------------------------------------------
# Agent message callback
# ------------------------------------------------------------------

def make_message_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""

    def _message(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_message_text(text)
        _send_update(conn, session_id, loop, update)

    return _message
