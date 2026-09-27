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
import threading
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    _json_loads_maybe,
    _text as _tool_text,
    build_tool_abandoned,
    build_tool_complete,
    build_tool_start,
    coerce_tool_args,
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
    if isinstance(result, str):
        if not result.strip():
            return None
        try:
            data = _json_loads_maybe_prefix(result)
        except Exception:
            return None
    elif isinstance(result, dict):
        data = result
    else:
        return None

    if not isinstance(data, dict):
        return None

    todos = next(
        (data[key] for key in ("todos", "todo_list", "todo") if isinstance(data.get(key), list)),
        None,
    )
    if todos is None:
        return None
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
) -> bool:
    """Schedule an ACP update without blocking; return whether scheduling succeeded."""
    try:
        update_coro = conn.session_update(session_id, update)
    except Exception:
        logger.debug("Failed to create ACP update", exc_info=True)
        return False
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is loop:
        # STOP runs on the ACP loop and synchronously interrupts required
        # children. Their terminal callbacks also land here. Waiting on
        # run_coroutine_threadsafe(...).result() from that same loop deadlocks
        # until the 5s timeout for every child. Schedule in-order on the loop
        # and observe failure asynchronously instead.
        task = loop.create_task(update_coro)

        def _observe_task(done: asyncio.Task) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                logger.debug("ACP update task cancelled")
            except Exception:
                logger.debug("Failed to send ACP update", exc_info=True)

        task.add_done_callback(_observe_task)
        return True

    # Worker-thread callbacks must also stay nonblocking. Some lifecycle
    # paths hold child ordering locks while scheduling an update; waiting for
    # the ACP loop here can invert that lock against STOP/interrupt delivery.
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        update_coro,
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return False

    def _observe_future(done) -> None:
        try:
            done.result()
        except Exception:
            logger.debug("Failed to send ACP update", exc_info=True)

    future.add_done_callback(_observe_future)
    return True


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def _upgrade_queue(tool_call_ids: Dict[str, Deque[str]], name: str) -> Deque[str] | None:
    """Fetch the per-tool FIFO of pending call IDs, upgrading a legacy bare-string entry in place."""
    queue = tool_call_ids.get(name)
    if isinstance(queue, str):
        queue = tool_call_ids[name] = deque([queue])
    return queue


def flush_open_tool_calls(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> int:
    """Close every tool call still open at the end of a turn, and report how many there were.

    A tool blocked by scope, guardrail or an editor permission prompt never
    projects ``tool.completed``, so without this its bubble stays ``in_progress``
    forever and clients read the turn as one that never ran a tool."""
    open_calls = [(name, list(queue)) for name, queue in list(tool_call_ids.items()) if queue]
    flushed = 0
    for name, ids in open_calls:
        for tc_id in ids:
            tool_call_meta.pop(tc_id, None)
            _send_update(conn, session_id, loop, build_tool_abandoned(tc_id, name))
            flushed += 1
        tool_call_ids.pop(name, None)
    if flushed:
        logger.debug("Flushed %d ACP tool call(s) left open at turn end", flushed)
    return flushed


def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    turn_state: Dict[str, Any] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and immediately pairs
    ordinary ``tool.completed`` events by their stable tool-call ID. The FIFO
    remains as a compatibility fallback for producers without stable IDs and
    for ``delegate_task``, whose completion has separate background semantics.
    ``turn_state["saw_completion"]`` stands the step-callback fallback down
    once a completion has been handled here.
    """

    # Per-child delegate_task subagent calls (subagent_id -> ACP tool call id)
    # plus the last relayed activity snippet per child, folded into the final
    # completion frame. Children run on worker threads; per-key dict ops are
    # safe under the GIL and each child only touches its own key.
    child_calls: Dict[str, str] = {}
    child_last_activity: Dict[str, str] = {}
    child_early_terminals: Dict[str, tuple[str, Dict[str, Any]]] = {}
    child_lock = threading.Lock()

    def _send_child_terminal(
        sid: str,
        tc_id: str,
        preview: str,
        kwargs: Dict[str, Any],
        last_activity: str | None,
    ) -> None:
        status_raw = str(kwargs.get("status") or "completed").strip().lower()
        status = (
            "completed"
            if status_raw in {"completed", "complete", "success", "done", ""}
            else "failed"
        )
        parts = []
        supervision_status = str(
            kwargs.get("supervision_status") or ""
        ).strip().lower()
        if supervision_status:
            parts.append(
                json.dumps(
                    {
                        "type": "subagent.supervision",
                        "subagentId": sid,
                        "supervisionStatus": supervision_status,
                    },
                    ensure_ascii=False,
                )
            )
        for candidate in (preview, kwargs.get("summary"), last_activity):
            text_value = str(candidate or "").strip()
            if text_value and text_value not in parts:
                parts.append(text_value)
        update = acp.update_tool_call(
            tc_id,
            kind="execute",
            status=status,
            content=[_tool_text("\n\n".join(parts))] if parts else None,
            raw_output=(
                {
                    "subagentId": sid,
                    "supervisionStatus": supervision_status,
                }
                if supervision_status
                else None
            ),
        )
        _send_update(conn, session_id, loop, update)

    def _handle_subagent_event(
        event_type: str, tool_name: str, preview: str, kwargs: Dict[str, Any]
    ) -> None:
        """Translate relayed ``subagent.*`` events into per-child ACP frames.

        delegate_tool relays child lifecycle fully identity-tagged
        (subagent_id/task_index/goal/...). Emission contract for ACP clients:
        one ToolCallStart on ``subagent.start``, same-id in-progress heartbeat
        updates, and one terminal update on ``subagent.complete``. Switchboard
        preserves same-id ``in_progress`` updates and closes only on terminal
        status.
        """
        if not _subagent_updates_enabled():
            return
        sid = kwargs.get("subagent_id")
        if sid is None:
            return
        sid = str(sid)

        if event_type == "subagent.start":
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
            terminal = None
            with child_lock:
                if sid in child_calls:
                    return
                tc_id = make_tool_call_id()
                child_calls[sid] = tc_id
                update = acp.start_tool_call(
                    tc_id,
                    title,
                    kind="execute",
                    raw_input={"tool": "subagent", "arguments": raw_arguments},
                )
                # Preserve wire order against a controller timeout racing this
                # start: the start is scheduled before a terminal can claim it.
                _send_update(conn, session_id, loop, update)
                pending = child_early_terminals.pop(sid, None)
                if pending is not None:
                    child_calls.pop(sid, None)
                    terminal = (
                        tc_id,
                        pending[0],
                        pending[1],
                        child_last_activity.pop(sid, None),
                    )
            if terminal is not None:
                _send_child_terminal(
                    sid,
                    terminal[0],
                    terminal[1],
                    terminal[2],
                    terminal[3],
                )
            return

        if event_type == "subagent.heartbeat":
            heartbeat = {
                "type": "subagent.heartbeat",
                "subagentId": sid,
                "elapsedSeconds": kwargs.get("elapsed_seconds"),
                "currentTool": kwargs.get("current_tool"),
                "lastActivity": kwargs.get("last_activity") or preview,
                "progressGeneration": kwargs.get("progress_generation", 0),
                "supervisionStatus": kwargs.get(
                    "supervision_status", "running"
                ),
            }
            with child_lock:
                tc_id = child_calls.get(sid)
                if tc_id is None:
                    return
                update = acp.update_tool_call(
                    tc_id,
                    kind="execute",
                    status="in_progress",
                    content=[_tool_text(json.dumps(heartbeat, ensure_ascii=False))],
                )
                # Keep an in-progress heartbeat from overtaking a terminal
                # update claimed concurrently on another worker thread.
                _send_update(conn, session_id, loop, update)
            return

        if event_type == "subagent.complete":
            with child_lock:
                tc_id = child_calls.pop(sid, None)
                last_activity = child_last_activity.pop(sid, None)
                if (
                    tc_id is None
                    and kwargs.get("supervision_terminal")
                    and sid not in child_early_terminals
                ):
                    # STOP can win before the worker emits subagent.start.
                    # Remember the controller terminal so a late start is
                    # immediately paired and can never strand an open card.
                    child_early_terminals[sid] = (preview, dict(kwargs))
            if tc_id is None:
                return
            _send_child_terminal(sid, tc_id, preview, kwargs, last_activity)
            return

        # subagent.tool / subagent.thinking / subagent.text / subagent.progress:
        # keep the freshest snippet for the completion frame.
        snippet = str(preview or tool_name or "").strip()
        if snippet:
            with child_lock:
                if sid in child_calls:
                    child_last_activity[sid] = snippet[:500]

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        if isinstance(event_type, str) and event_type.startswith("subagent."):
            _handle_subagent_event(event_type, name, preview, kwargs)
            return
        if event_type == "tool.completed" and name:
            if turn_state is not None:
                turn_state["saw_completion"] = True
            # delegate_task's completion has separate background/async dispatch-card
            # semantics (see tools.py's build_async_background_completion); it never
            # closes through this immediate fast path.
            if name == "delegate_task":
                return
            queue = _upgrade_queue(tool_call_ids, name)
            source_id = str(kwargs.get("tool_call_id") or "").strip()
            tc_id = None
            if queue:
                if source_id:
                    if source_id in queue:
                        tc_id = source_id
                else:
                    tc_id = queue[0]
            if tc_id is None:
                logger.debug(
                    "ACP completion for %r has no matching tool_call id; dropping",
                    name,
                )
                return
            meta = tool_call_meta.get(tc_id, {})
            result = kwargs.get("result")
            try:
                # The executor's verdict: a cancelled/errored tool may return plain text the heuristic misses.
                update = build_tool_complete(
                    tc_id,
                    name,
                    result=result,
                    function_args=meta.get("args"),
                    snapshot=meta.get("snapshot"),
                    is_error=bool(kwargs.get("is_error")),
                )
            except Exception:
                # Leave the call tracked so the end-turn flush can still close
                # its card with a minimal terminal update.
                logger.debug(
                    "Failed to render ACP completion for %r", name, exc_info=True
                )
                return
            if not _send_update(conn, session_id, loop, update):
                return
            queue.remove(tc_id)
            tool_call_meta.pop(tc_id, None)
            if name in {"todo", "todo_list"}:
                plan_update = _build_plan_update_from_todo_result(result)
                if plan_update is not None:
                    _send_update(conn, session_id, loop, plan_update)
            if not queue:
                tool_call_ids.pop(name, None)
            return
        # Only emit ACP ToolCallStart for tool.started; ignore other event types.
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = str(kwargs.get("tool_call_id") or "").strip() or make_tool_call_id()
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
# Assistant message identity
# ------------------------------------------------------------------


class AssistantMessageIdAllocator:
    """Allocates stable per-message ids for streamed assistant chunks.

    ACP clients group streamed ``agent_message_chunk`` / ``agent_thought_chunk``
    deltas into one assistant reply by ``messageId`` and use a NEW id to start
    the next reply (root-reply replacement semantics). Without ids, a client
    that replaces "the current assistant message" on each chunk collapses
    separate autonomous turns into one bubble.

    One allocator lives per ACP session; a contiguous run of deltas shares
    ``current()`` and ``close()`` marks the message finished so the next delta
    allocates a fresh id. Ids are UUID4 strings because the ACP schema requires
    UUID-format message ids, and a fresh UUID can never collide with an earlier
    turn's id.
    """

    def __init__(self) -> None:
        self._active: str | None = None
        self._last: str | None = None

    def current(self) -> str:
        """Return the active message id, allocating one if none is open."""
        if self._active is None:
            self._active = self._last = str(uuid.uuid4())
        return self._active

    def last(self) -> str | None:
        """Return the most recently allocated id (open or closed)."""
        return self._last

    def close(self) -> None:
        """End the active message; the next chunk starts a new id."""
        self._active = None


def _make_text_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, wrap: Callable[[str], Any],
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    # ``None`` is the flush sentinel Hermes core sends between assistant messages
    # (before tool execution / at end of stream): it closes the active messageId so
    # the next delta opens a new bubble instead of merging into the previous one.
    def _cb(text: str | None) -> None:
        if text:
            update = wrap(text)
            if message_ids is not None:
                update.message_id = message_ids.current()
            _send_update(conn, session_id, loop, update)
        elif text is None and message_ids is not None:
            message_ids.close()

    return _cb


def make_thinking_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_thought_text, message_ids)


def make_message_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_message_text, message_ids)


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    turn_state: Dict[str, Any] | None = None,
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if not isinstance(prev_tools, list):
            return
        for tool_info in prev_tools:
            tool_name = result = function_args = None
            if isinstance(tool_info, dict):
                tool_name = tool_info.get("name") or tool_info.get("function_name")
                # Key presence, not truthiness: "", 0 and False are real results (#10845).
                result = tool_info.get("result") if "result" in tool_info else tool_info.get("output")
                function_args = tool_info.get("arguments") or tool_info.get("args")
            elif isinstance(tool_info, str):
                tool_name = tool_info

            if not tool_name:
                continue
            # ``tool.completed`` already closed this call with its own result;
            # this callback is the fallback for runtimes that never project one.
            if not (turn_state or {}).get("saw_completion"):
                queue = _upgrade_queue(tool_call_ids, tool_name)
                if queue:
                    tc_id = queue.popleft()
                    meta = tool_call_meta.pop(tc_id, {})
                    # ``prev_tools`` carries the wire ``arguments`` JSON *string*; the content
                    # builders index it as a dict, so an uncoerced string raised inside this
                    # (swallowed) callback and the bubble never closed.
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=(
                            result
                            if isinstance(result, str)
                            else json.dumps(result, ensure_ascii=False)
                            if isinstance(result, (dict, list))
                            else str(result)
                            if result is not None
                            else None
                        ),
                        function_args=coerce_tool_args(function_args) if function_args else meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)
                else:
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
            if tool_name in {"todo", "todo_list"}:
                plan_update = _build_plan_update_from_todo_result(result)
                if plan_update is not None:
                    _send_update(conn, session_id, loop, plan_update)

    return _step
