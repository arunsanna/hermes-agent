"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import contextlib
import contextvars
import logging
import os
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities, AgentMessageChunk, AuthenticateResponse, ClientCapabilities, ForkSessionResponse,
    Implementation, InitializeResponse, ListSessionsResponse, LoadSessionResponse, McpServerHttp, McpServerSse,
    McpServerStdio, ModelInfo, NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionForkCapabilities, SessionInfo, SessionInfoUpdate, SessionListCapabilities,
    SessionMode, SessionModeState, SessionModelState, SessionResumeCapabilities, SetSessionConfigOptionResponse,
    SetSessionModeResponse, SetSessionModelResponse, TextContentBlock, Usage, UsageUpdate, UserMessageChunk,
)

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider
from acp_adapter.commands import (
    HERMES_VERSION, RunPromptAfterCommand, SlashCommandsMixin, _estimate_tokens, _get_goal_manager,
)
from acp_adapter.content import PromptBlock, _content_blocks_to_openai_user_content, _extract_text
from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    flush_open_tool_calls,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.model_catalog import build_model_state, encode_model_choice
from acp_adapter.permissions import make_approval_callback
from acp_adapter.provenance import session_provenance_meta
from acp_adapter.session import (
    OwnedSessions,
    SessionManager,
    SessionState,
    UnsafeSessionTranscriptError,
    _expand_acp_enabled_toolsets,
)
from acp_adapter.tools import (
    _async_background_delegation_id,
    build_async_background_completion,
    build_tool_complete,
    build_tool_start,
    flush_async_background_dispatches,
    coerce_tool_args,
)
from agent.context_compressor import (COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor)
from agent.interrupt_compat import request_hard_interrupt
from tools.approval_context import reset_hermes_interactive_context, set_hermes_interactive_context

logger = logging.getLogger(__name__)

def _dispose_replaced_agent(agent: Any) -> None:
    """Release replacement-owned clients without touching shared session state."""
    if agent is None:
        return
    try:
        attrs = getattr(agent, "__dict__", {})
        codex_session = attrs.get("_codex_session") if isinstance(attrs, dict) else None
        if codex_session is not None:
            codex_session.close()
            agent._codex_session = None
    except Exception:
        logger.debug("Failed to close replaced Codex app-server session", exc_info=True)
    try:
        release_clients = getattr(agent, "release_clients", None)
        if callable(release_clients):
            release_clients()
    except Exception:
        logger.debug("Failed to release replaced ACP agent clients", exc_info=True)

# Runs the synchronous AIAgent off the event loop.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# ListSessionsRequest has no client-side limit; clients paginate via `cursor`/`next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50
_TURN_KEEPALIVE_INTERVAL_DEFAULT = 45.0
_TURN_KEEPALIVE_MAX_SILENT_DEFAULT = 1800.0


def _turn_keepalive_settings() -> tuple[float, float]:
    """(interval_seconds, max_silent_seconds) for the in-turn ACP keepalive.

    interval <= 0 disables the keepalive entirely. max_silent bounds how long
    the loop will vouch for an agent that shows no internal liveness touches;
    past it the loop goes quiet so the gateway stall watchdog can reclaim a
    genuinely wedged turn. Config keys (config.yaml):
    ``acp.turn_keepalive_interval_seconds`` and
    ``acp.turn_keepalive_max_silent_seconds``.
    """
    interval = _TURN_KEEPALIVE_INTERVAL_DEFAULT
    max_silent = _TURN_KEEPALIVE_MAX_SILENT_DEFAULT
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        acp_cfg = cfg.get("acp") if isinstance(cfg, dict) else None
        if isinstance(acp_cfg, dict):
            interval = float(
                acp_cfg.get("turn_keepalive_interval_seconds", interval)
            )
            max_silent = float(
                acp_cfg.get("turn_keepalive_max_silent_seconds", max_silent)
            )
    except Exception:
        pass
    return interval, max_silent

# JSON-RPC 2.0 reserves -32000..-32099 for implementation-defined server
# errors (acp.exceptions.RequestError already uses -32000 for auth_required
# and -32002 for resource_not_found). This is the cross-session ownership
# guard's own code in that same reserved band.
_CROSS_SESSION_GUARD_ERROR_CODE = -32001

def _sanitize_failed_turn_history(
    messages: Any,
    *,
    baseline_count: int,
) -> list[dict[str, Any]]:
    """Remove current-turn visible assistant candidates, preserving protocol."""
    if not isinstance(messages, list):
        return []

    sanitized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            sanitized.append(dict(message))

    start = min(max(int(baseline_count), 0), len(sanitized))
    if baseline_count > len(sanitized):
        # Compression may have replaced the prefix. In that case the latest
        # user message is the only safe current-turn boundary available.
        for index, message in enumerate(sanitized):
            if message.get("role") == "user":
                start = index + 1

    from agent.conversation_loop import (
        _sanitize_required_assistant_candidate,
    )

    output = sanitized[:start]
    for message in sanitized[start:]:
        if message.get("role") != "assistant":
            output.append(message)
            continue
        _sanitize_required_assistant_candidate(message)
        has_protocol = any(
            message.get(key)
            for key in (
                "tool_calls",
                "function_call",
                "codex_reasoning_items",
                "anthropic_content_blocks",
                "reasoning",
                "reasoning_content",
            )
        )
        if has_protocol:
            output.append(message)
        # A plain assistant candidate is removed entirely. Keeping an empty
        # assistant row would still alter role sequencing on resume.
    return output


def _rewrite_agent_active_history(
    agent: Any,
    messages: list[dict],
    state: SessionState,
    session_manager: SessionManager,
) -> bool:
    """Correct an already-flushed active transcript after a fail-closed exit.

    ``replace_messages`` is atomic, so retrying a transient write failure cannot
    expose a partial transcript. The boolean result is deliberately observable:
    callers must not report a clean terminal outcome when durable correction
    never succeeded.

    Poison-first: the durable safety marker is written BEFORE the correction
    below is attempted, not only recorded once every retry has already
    failed. The candidate this function exists to correct was already
    flushed to state.db by the turn that triggered this call (see the
    docstring above — "an already-flushed active transcript"). A process
    kill landing between that flush and this function's own
    ``replace_messages`` call previously left no durable trace at all, so a
    fresh ``_restore`` would silently reload the tainted row (content AND
    any api_content sidecar) as if nothing had gone wrong — defeating the
    fail-closed design for exactly the crash window it exists to cover.
    Marking poisoned first closes that window: any crash from this point
    forward leaves the marker set, so ``_restore`` refuses resume
    (raises ``UnsafeSessionTranscriptError``) until the successful-rewrite
    branch below durably clears it again.
    """
    agent._session_messages = messages
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if db is None or not session_id:
        return True

    if not session_manager.mark_transcript_correction_poisoned(state):
        logger.critical(
            "Failed to persist transcript safety marker for ACP session %s "
            "before attempting active transcript correction; retrying the "
            "marker write after the correction attempt below",
            session_id,
        )

    try:
        for attempt in range(1, 4):
            try:
                db.replace_messages(
                    session_id,
                    # ``replace_messages`` annotates inserted dicts with their
                    # fresh SQLite row ids. Keep those transport-internal ids
                    # out of the ACP working history: restore normally uses
                    # ``include_row_ids=False`` and a correction must not
                    # change that transcript shape merely by persisting it.
                    deepcopy(messages),
                    # Fail-closed correction targets only the live turn. Never
                    # risk deleting compacted/undo history while removing a
                    # rejected active candidate.
                    active_only=True,
                )
                agent._flushed_db_message_ids = set()
                agent._flushed_db_message_session_id = session_id
                agent._last_flushed_db_idx = len(messages)
                # Poison-first guarantees transcript_correction_poisoned is
                # already True here (set unconditionally, in-memory, by the
                # proactive mark above regardless of whether it durably
                # persisted) — always attempt to clear it on success. A
                # transient clear failure alone must not permanently brick an
                # already-corrected, healthy session (the transcript itself
                # is genuinely clean at this point — replace_messages just
                # succeeded): retry with the same bounded-attempts, no-sleep
                # convention as the replace_messages retry immediately above,
                # rather than inventing a new one. Even if every attempt here
                # fails, the marker is a redo flag, not a tombstone — a
                # future restore's self-heal (acp_adapter.session._restore)
                # re-attempts this exact sanitize+rewrite+clear sequence and
                # is idempotent when the transcript is already clean.
                cleared = False
                for clear_attempt in range(1, 4):
                    if session_manager.clear_transcript_correction_poisoned(
                        state
                    ):
                        cleared = True
                        break
                    if clear_attempt < 3:
                        logger.warning(
                            "Retrying transcript safety marker clear for %s "
                            "after attempt %s",
                            session_id,
                            clear_attempt,
                        )
                if not cleared:
                    logger.error(
                        "Corrected active transcript for %s but could not "
                        "durably clear its safety marker after %s attempts; "
                        "session remains poisoned until a future resume "
                        "self-heals it",
                        session_id,
                        3,
                    )
                    return False
                return True
            except Exception:
                if attempt == 3:
                    logger.exception(
                        "Failed to correct active transcript after ACP fail-close "
                        "for %s after %s attempts",
                        session_id,
                        attempt,
                    )
                else:
                    logger.warning(
                        "Retrying active transcript correction for %s after "
                        "attempt %s",
                        session_id,
                        attempt,
                        exc_info=True,
                    )
    except Exception:
        logger.exception(
            "Failed to prepare active transcript correction after ACP "
            "fail-close for %s",
            session_id,
        )
    marker_persisted = session_manager.mark_transcript_correction_poisoned(
        state
    )
    if not marker_persisted:
        logger.critical(
            "Failed to persist transcript safety marker for ACP session %s; "
            "durable replay safety cannot be guaranteed",
            session_id,
        )
    return False


def _flatten_history_text(value: Any) -> str:
    """Persisted content/reasoning (str, or list of ``{"text"}`` / ``{"type": "text", "content"}``
    parts) -> one stripped string; whitespace-only collapses to ``""`` ("nothing to emit")."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def _history_reasoning_text(message: dict[str, Any]) -> str:
    """First non-empty of ``reasoning_content`` and ``reasoning`` — both live keys, for
    different transports (not old-vs-new)."""
    for key in ("reasoning_content", "reasoning"):
        text = _flatten_history_text(message.get(key))
        if text:
            return text
    return ""


def _history_summary_meta(message: dict[str, Any], text: str) -> dict[str, Any] | None:
    """``_meta`` for a replayed compaction summary, else None.

    Summaries persist as ordinary messages, standalone (either role) or merged into the first
    preserved tail message. Two keys so clients can't hide real content: ``compactionSummary``
    (whole chunk; safe to collapse) vs ``containsCompactionSummary`` (real content + summary).
    Uses the in-process flag, falling back to content classification for DB-reloaded sessions."""
    kind = ContextCompressor.classify_summary_content(text)
    if kind is None and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
        # Flagged but unclassified (prefix drift): the flag only marks summaries -> standalone.
        kind = "standalone"
    if kind == "standalone":
        return {"hermes": {"compactionSummary": True}}
    if kind == "merged":
        return {"hermes": {"containsCompactionSummary": True}}
    return None


# role -> (chunk class, session_update tag) for history replay.
_HISTORY_CHUNK_TYPES = {
    "user": (UserMessageChunk, "user_message_chunk"), "assistant": (AgentMessageChunk, "agent_message_chunk")
}


def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract function name/arguments from an OpenAI-style tool_call."""
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
    raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
    return name, coerce_tool_args(raw_args)


def _history_message_chunk(role: str, message: dict[str, Any]) -> UserMessageChunk | AgentMessageChunk | None:
    text = _flatten_history_text(message.get("content"))
    if not text:
        return None
    cls, session_update = _HISTORY_CHUNK_TYPES[role]
    return cls(
        session_update=session_update, content=TextContentBlock(type="text", text=text),
        field_meta=_history_summary_meta(message, text),
    )


def _history_replay_updates(history: list[dict[str, Any]]):
    """Yield ACP session updates that reconstruct a persisted transcript, in order: user/assistant
    text (with compaction ``_meta``), assistant thoughts, and tool-call start/complete pairs
    (``todo`` results also re-emit the plan)."""
    active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in history:
        role = str(message.get("role") or "")
        if role == "user":
            if (chunk := _history_message_chunk(role, message)) is not None:
                yield chunk
        elif role == "assistant":
            thought = _history_reasoning_text(message)
            if thought:
                yield acp.update_agent_thought_text(thought)
            if (chunk := _history_message_chunk(role, message)) is not None:
                yield chunk
            tool_calls = message.get("tool_calls")
            for tool_call in tool_calls if isinstance(tool_calls, list) else ():
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = str(
                    tool_call.get("id") or tool_call.get("call_id") or tool_call.get("tool_call_id") or ""
                ).strip()
                if not tool_call_id:
                    continue
                tool_name, args = _history_tool_call_name_args(tool_call)
                active_tool_calls[tool_call_id] = (tool_name, args)
                yield build_tool_start(tool_call_id, tool_name, args)
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            tool_name = str(message.get("tool_name") or "").strip()
            function_args: dict[str, Any] | None = None
            if tool_call_id in active_tool_calls:
                tool_name, function_args = active_tool_calls.pop(tool_call_id)
            if not tool_call_id or not tool_name:
                continue
            result = message.get("content")
            result_text = result if isinstance(result, (str, dict, list)) else None
            yield build_tool_complete(tool_call_id, tool_name, result=result_text, function_args=function_args)
            if tool_name in {"todo", "todo_list"}:
                plan_update = _build_plan_update_from_todo_result(result_text)
                if plan_update is not None:
                    yield plan_update


def _mcp_server_config(server: McpServerStdio | McpServerHttp | McpServerSse) -> dict:
    if isinstance(server, McpServerStdio):
        return {"command": server.command, "args": list(server.args), "env": {i.name: i.value for i in server.env}}
    return {"url": server.url, "headers": {i.name: i.value for i in server.headers}}


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _bind_guarded(stack: contextlib.ExitStack, label: str, setup: Callable[[], Callable[[], None]]) -> None:
    """Run ``setup`` (returns its teardown) and register the teardown; failures in either half only
    log — the turn must still run without the binding."""
    try:
        teardown = setup()
    except Exception:
        logger.debug("Could not set ACP %s", label, exc_info=True)
        return

    def _teardown() -> None:
        try:
            teardown()
        except Exception:
            logger.debug("Could not restore ACP %s", label, exc_info=True)

    stack.callback(_teardown)


def _attach_interrupted_prompt(interrupted_prompt: str, guidance: str) -> str:
    return f"{interrupted_prompt}\n\nUser correction/guidance after interrupt: {guidance}"


def _take_interrupted_prompt(state: SessionState) -> tuple[bool, str]:
    """``(idle, interrupted_prompt)``; consumes the cancelled prompt only when the session is idle."""
    with state.runtime_lock:
        if state.is_running:
            return False, ""
        text, state.interrupted_prompt_text = state.interrupted_prompt_text, ""
        return True, text


@dataclass
class _TurnCallbacks:
    """Per-turn ACP streaming callbacks; all None when no client is connected."""

    tool_progress_cb: Any = None
    reasoning_cb: Any = None
    step_cb: Any = None
    stream_delta_cb: Any = None
    approval_cb: Any = None
    edit_approval_requester: Any = None
    streamed: bool = False


class HermesACPAgent(SlashCommandsMixin, acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"
    # mode id -> (edit approval policy, display name, description)
    _MODES: dict[str, tuple[str, str, str]] = {
        "default": ("ask", "Default", "Ask before edits."),
        "accept_edits": (
            "workspace_session",
            "Accept Edits",
            "Auto-allow workspace and /tmp edits; still asks for sensitive paths.",
        ),
        "dont_ask": (
            "session", "Don't Ask", "Auto-allow file edits for this session except sensitive paths."
        ),
    }
    _MODE_TO_EDIT_APPROVAL_POLICY = {mode: spec[0] for mode, spec in _MODES.items()}
    _EDIT_APPROVAL_POLICY_TO_MODE = {spec[0]: mode for mode, spec in _MODES.items()}

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None
        self._delegation_watcher_task: Optional[asyncio.Task] = None

    def _ensure_delegation_watcher(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the completion watcher once, on the running loop."""
        if self._delegation_watcher_task is not None:
            return
        if os.environ.get("HERMES_ACP_BACKGROUND_COMPLETIONS", "1").strip().lower() in {
            "0",
            "false",
            "off",
        }:
            return
        self._delegation_watcher_task = loop.create_task(self._async_delegation_watcher())

    async def _async_delegation_watcher(self, interval: float = 2.0) -> None:
        """Drain background-delegation completions back into their sessions.

        ``delegate_task(background=true)`` results land on the shared
        ``process_registry.completion_queue``, which nothing in the ACP
        process consumed — a background child finishing after its turn ended
        was silently lost. Mirrors gateway/run.py's watcher: route by
        ``session_key`` (== the ACP session id, bound via ``set_session_vars``
        during ``prompt()``), append the formatted notification to session
        history so the agent sees it on its next turn, and emit an
        out-of-turn tool frame pair so ACP clients render it immediately
        (Switchboard shows these as a continuation turn).

        Injection is deferred while a turn is running: ``prompt()`` REPLACES
        ``state.history`` with the agent's result at turn end, so an append
        mid-turn would be lost — busy-session events are requeued and retried
        on the next tick.
        """
        try:
            from tools.process_registry import (
                format_process_notification,
                process_registry as _pr,
            )
        except Exception:
            logger.exception("Delegation watcher disabled: process_registry unavailable")
            return
        while True:
            try:
                await self._drain_completion_queue_once(_pr, format_process_notification)
            except Exception:
                logger.exception("Async delegation watcher error")
            await asyncio.sleep(interval)

    async def _drain_completion_queue_once(self, pr, formatter) -> None:
        """One watcher tick: route completions, requeue what isn't ours yet.

        Two ownership gates, both required — see #delegation-cross-session-leak
        (2026-07-25 Switchboard incident: a delegation dispatched under one
        session was re-delivered into four unrelated sessions):

        1. ``peek_session`` (never ``get_session``) — this watcher may only
           act on a session already resident IN THIS PROCESS. ``get_session``
           transparently restores from the SessionDB on a miss, which is
           correct for an explicit ``session/load`` from the editor but wrong
           here: when multiple ACP processes share one SessionDB (a host that
           doesn't isolate HERMES_HOME per process), it would silently adopt
           a stranger's session, splice the completion into their history,
           persist it back to the shared DB, and broadcast a ``session/update``
           notification for their session_id over THIS process's connection.
        2. ``claim_event_delivery`` — a completion may still be
           ``delivery_state='pending'`` in a SessionDB shared across
           processes/hosts (nothing here marks it delivered on its own), so
           without this claim every process that boots and calls
           ``restore_undelivered_completions`` would re-inject the same
           durable event into its owning session forever. Matches the
           claim/complete/release pattern every other completion_queue
           consumer in the codebase already uses (``tui_gateway/server.py``,
           ``cli.py``).
        """
        from tools.async_delegation import (
            claim_event_delivery,
            complete_event_delivery,
            release_event_delivery,
        )

        drained = []
        while not pr.completion_queue.empty():
            try:
                drained.append(pr.completion_queue.get_nowait())
            except Exception:
                break

        requeue = []
        for evt in drained:
            if evt.get("type") != "async_delegation":
                # Owned by other drain patterns (watch events etc.); keep
                # them available like the gateway watcher does.
                requeue.append(evt)
                continue
            session_id = str(evt.get("session_key") or "")
            state = self.session_manager.peek_session(session_id) if session_id else None
            if state is None:
                logger.warning(
                    "Dropping async delegation %s: no live ACP session for "
                    "key %r in this process",
                    evt.get("delegation_id"),
                    session_id,
                )
                continue
            with state.runtime_lock:
                busy = state.is_running
            if busy:
                requeue.append(evt)
                continue
            claim = claim_event_delivery(evt, "acp-watcher")
            if claim is None:
                # Another process/consumer already claimed (or delivered)
                # this durable completion — do not duplicate it here.
                continue
            try:
                with state.runtime_lock:
                    text = formatter(evt)
                    if text:
                        state.history.append({"role": "user", "content": text})
                self.session_manager.save_session(session_id)
                await self._notify_background_completion(session_id, evt)
            except Exception:
                release_event_delivery(evt, claim)
                raise
            else:
                complete_event_delivery(evt, claim)
        for evt in requeue:
            pr.completion_queue.put(evt)

    async def _drain_queued_prompts(self, state, session_id, conn):
        """Run every prompt the caller queued while this turn was in flight,
        as normal follow-up user turns (preserving role alternation and
        history). Every `prompt()` return path — normal completion, a
        cancelled turn, a post-barrier exception, or an exception from the
        initial executor dispatch itself — must call this exactly once after
        freeing `is_running`, so a queued follow-up never sits stuck (Phase 0
        / stop-p0-brief.md P0.1, HOLE 2)."""
        while True:
            with state.runtime_lock:
                if not state.queued_prompts:
                    break
                next_prompt = state.queued_prompts.pop(0)
            if conn:
                await conn.session_update(
                    session_id,
                    acp.update_user_message_text(next_prompt),
                )
            try:
                await self.prompt(
                    prompt=[TextContentBlock(type="text", text=next_prompt)],
                    session_id=session_id,
                )
            except acp.RequestError as exc:
                # A drained follow-up is a synthesized turn with no live
                # incoming JSON-RPC request of its own — there is nothing to
                # attach an error response to, and letting it propagate would
                # misattribute the failure to the (possibly already
                # successful) outer request whose finally-block ran this
                # drain. Surface the failure text in-band instead.
                logger.error(
                    "Queued follow-up prompt failed for session %s: %s",
                    session_id,
                    exc,
                )
                if conn:
                    try:
                        await conn.session_update(
                            session_id,
                            acp.update_agent_message_text(str(exc)),
                        )
                    except Exception:
                        logger.exception(
                            "Could not deliver queued-prompt failure notice "
                            "for session %s",
                            session_id,
                        )

    @staticmethod
    def _drain_session_delegation_completions(pr, formatter, session_id, state):
        """Take this session's delegation events and requeue everything else.

        The ``session_key`` match below is inherently self-scoped (``state``
        is this call's own live session — there is no foreign-session
        adoption risk here, unlike ``_drain_completion_queue_once``). Still
        claim before delivering: without it, a completion whose
        ``delivery_state`` is still ``pending`` in the SessionDB would be
        re-spliced into this SAME session's history every time this process
        (or a future one restoring the same durable row) re-drains the
        queue, e.g. after a restart.
        """
        from tools.async_delegation import claim_event_delivery, complete_event_delivery

        drained = []
        while not pr.completion_queue.empty():
            try:
                drained.append(pr.completion_queue.get_nowait())
            except Exception:
                break

        matched = []
        requeue = []
        for evt in drained:
            if (
                evt.get("type") != "async_delegation"
                or str(evt.get("session_key") or "") != session_id
            ):
                requeue.append(evt)
                continue
            claim = claim_event_delivery(evt, "acp-join")
            if claim is None:
                # Already delivered by another consumer (e.g. the background
                # watcher won the race first) — do not duplicate it here.
                continue
            matched.append(evt)
            text = formatter(evt)
            if text:
                state.history.append({"role": "user", "content": text})
            complete_event_delivery(evt, claim)

        for evt in requeue:
            pr.completion_queue.put(evt)
        return matched

    async def _notify_background_completion(self, session_id: str, evt: dict) -> None:
        """Emit an out-of-turn tool frame pair for a finished background child."""
        conn = self._conn
        if conn is None:
            return
        from acp_adapter.tools import _text as _tool_text, make_tool_call_id

        goal = str(evt.get("goal") or "").strip()
        status_raw = str(evt.get("status") or "completed").strip().lower()
        ok = status_raw in {"completed", "complete", "success", "done", ""}
        title = "background delegation " + ("completed" if ok else status_raw)
        if goal:
            title += ": " + (goal[:100] + ("…" if len(goal) > 100 else ""))
        body = str(evt.get("summary") or evt.get("error") or "").strip()
        tc_id = make_tool_call_id()
        raw_arguments = {"background": True}
        if goal:
            raw_arguments["goal"] = goal[:400]
        if evt.get("delegation_id"):
            raw_arguments["delegationId"] = evt.get("delegation_id")
        if evt.get("model"):
            raw_arguments["model"] = evt.get("model")
        try:
            await conn.session_update(
                session_id,
                acp.start_tool_call(
                    tc_id,
                    title,
                    kind="execute",
                    raw_input={"tool": "subagent", "arguments": raw_arguments},
                ),
            )
            await conn.session_update(
                session_id,
                acp.update_tool_call(
                    tc_id,
                    kind="execute",
                    status="completed" if ok else "failed",
                    content=[_tool_text(body)] if body else None,
                ),
            )
        except Exception:
            logger.debug("Failed to send background completion frames", exc_info=True)
    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    async def _send(self, session_id: str, update: Any, *, fail_msg: str, level: int = logging.WARNING) -> bool:
        """``session_update`` that logs instead of raising; False on failure."""
        try:
            await self._conn.session_update(session_id=session_id, update=update)
            return True
        except Exception:
            logger.log(level, fail_msg, session_id, exc_info=True)
            return False

    def _schedule_soon(self, make_coro: Callable[[], Any]) -> None:
        """Run a notification coroutine right after the current response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, make_coro())

    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Edit-approval policy as ACP modes. Zed renders ``config_options`` in the model
        picker's slot; modes (as Claude/Codex use) coexist with the picker."""
        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODES:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[SessionMode(id=m, name=n, description=d) for m, (_p, n, d) in self._MODES.items()],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Authenticated providers + models, from the shared Hermes inventory (same substrate
        as ``hermes model``/TUI/dashboard) so the selector isn't just the current curated list."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"
        try:
            picker = build_model_state(model, provider, str(getattr(state.agent, "base_url", "") or ""))
            if picker is not None:
                return picker
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None
        choice = encode_model_choice(provider, model)
        return SessionModelState(available_models=[ModelInfo(model_id=choice, name=model)], current_model_id=choice)

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id."""
        target_provider, new_model = current_provider, raw_model.strip()
        try:
            from hermes_cli.models import detect_provider_for_model, parse_model_input

            raw_selection = new_model
            target_provider, new_model = parse_model_input(raw_selection, current_provider)
            # ``parse_model_input`` strips a recognized ``provider:`` prefix.
            # That explicit choice is authoritative even when it names the
            # current provider; auto-detection must not remap the model.
            has_explicit_provider = new_model != raw_selection
            if not has_explicit_provider and target_provider == current_provider:
                detected = detect_provider_for_model(new_model, current_provider)
                if detected:
                    target_provider, new_model = detected
        except Exception:
            logger.debug("Provider detection failed, using model as-is", exc_info=True)
        return target_provider, new_model

    def _commit_model_switch(self, state: SessionState, *, model: str, new_agent: Any) -> None:
        """Persist a replacement before retiring the previous live agent."""
        previous_model, previous_agent = state.model, state.agent
        state.model, state.agent = model, new_agent
        saved = self.session_manager.save_session(state.session_id)
        if saved is False:
            state.model, state.agent = previous_model, previous_agent
            self.session_manager.save_session(state.session_id)
            _dispose_replaced_agent(new_agent)
            raise RuntimeError("Failed to persist ACP model switch")
        _dispose_replaced_agent(previous_agent)

    def _switch_model(
        self, state: SessionState, raw_model: str, *, keep_endpoint: bool = False
    ) -> tuple[str | None, str, str]:
        """Rebuild the session agent on a new model -> (old provider, new provider, model).
        ``keep_endpoint`` carries base_url/api_mode over when the provider is unchanged."""
        current_provider = getattr(state.agent, "provider", None)
        target_provider, new_model = self._resolve_model_selection(raw_model, current_provider or "openrouter")
        endpoint: dict[str, Any] = {}
        if keep_endpoint and not (current_provider and target_provider != current_provider):
            endpoint = {
                "base_url": getattr(state.agent, "base_url", None), "api_mode": getattr(state.agent, "api_mode", None)
            }
        new_agent = self.session_manager._make_agent(
            session_id=state.session_id, cwd=state.cwd, model=new_model,
            requested_provider=target_provider, **endpoint,
        )
        self._commit_model_switch(state, model=new_model, new_agent=new_agent)
        return current_provider, target_provider, new_model

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """``usage_update`` for Zed's context indicator: ``size`` = context window, ``used`` =
        estimated request pressure (system prompt + history + tool schemas)."""
        compressor = getattr(state.agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None
        try:
            used = _estimate_tokens(state.history, state.agent)
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
        return UsageUpdate(session_update="usage_update", size=max(size, 0), used=max(used, 0))

    async def _send_usage_update(self, state: SessionState) -> None:
        if self._conn and (update := self._build_usage_update(state)) is not None:
            await self._send(state.session_id, update, fail_msg="Failed to send ACP usage update for session %s")

    async def _turn_keepalive_loop(self, state: SessionState) -> None:
        """Feed the gateway's turn-stall watchdog during long SILENT work.

        The Switchboard gateway force-closes a turn after
        HERMES_TURN_STALL_SECS (default 300s) with zero ``session/update``
        frames, but several healthy operations are wire-silent far longer:
        blocking LLM calls (reasoning TTFB floors reach 600s), MCP tools
        (300s default timeout), the compression summarizer (300s floor), and
        long single tool runs. None of them emit progress frames while they
        block, so the watchdog killed genuinely working turns at 5 minutes.

        This loop emits a schema-valid ``usage_update`` at a bounded rate
        while the agent still shows internal liveness
        (``agent._last_activity_ts``). usage_update is deliberate: the
        gateway treats it as non-content-bearing bookkeeping (resets the
        watchdog, opens no turn, renders nothing), and it keeps the client's
        context gauge fresh as a side benefit.

        Wedge detection stays intact: once the agent has recorded no
        liveness touch for ``max_silent`` seconds (default 30 min) the loop
        goes quiet — without unblocking — so a truly hung turn is still
        reclaimed by the gateway one stall window later. STOP-driven
        interrupt escalation is unaffected either way.
        """
        interval, max_silent = _turn_keepalive_settings()
        if interval <= 0:
            return
        agent = getattr(state, "agent", None)
        try:
            while True:
                await asyncio.sleep(interval)
                if not state.is_running:
                    return
                last = (
                    float(getattr(agent, "_last_activity_ts", 0.0) or 0.0)
                    if agent is not None
                    else 0.0
                )
                if last and (time.time() - last) > max_silent:
                    continue
                await self._send_usage_update(state)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "turn keepalive loop for session %s exited",
                getattr(state, "session_id", "?"),
                exc_info=True,
            )

    def _provenance_meta(
        self, acp_session_id: str, current_hermes_session_id: str, previous_hermes_session_id: Optional[str] = None
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(), acp_session_id, current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug("Could not build ACP session provenance for %s", acp_session_id, exc_info=True)
            return None

    def _session_meta(
        self,
        state: SessionState,
        *,
        current_hermes_session_id: Optional[str] = None,
        previous_hermes_session_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Merge standard provenance with verified orchestration evidence."""
        current_id = (
            current_hermes_session_id
            or getattr(state.agent, "session_id", state.session_id)
        )
        meta = self._provenance_meta(
            state.session_id,
            current_id,
            previous_hermes_session_id,
        ) or {}
        try:
            from acp_adapter.orchestration import orchestration_meta

            contract = orchestration_meta(state.agent)
            if contract is not None:
                meta["switchboardOrchestration"] = contract
        except Exception as exc:
            meta["switchboardOrchestration"] = {
                "requestedMode": os.environ.get("HERMES_ACP_ORCHESTRATION_MODE"),
                "effectiveMode": "single",
                "disabledToolsets": [],
                "effectiveTools": [],
                "mcpServers": [],
                "mcpRegistrationVerified": False,
                "verified": False,
                "mismatchReason": f"could not verify effective tool policy: {exc}",
            }
        return meta or None

    async def _send_session_info_update(
        self,
        session_id: str,
        *,
        current_hermes_session_id: Optional[str] = None,
        previous_hermes_session_id: Optional[str] = None,
    ) -> None:
        """Send ACP native session metadata after Hermes changes it."""
        if not self._conn:
            return
        try:
            row = self.session_manager._get_db().get_session(session_id)
        except Exception:
            logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
            return
        if not row:
            return
        title = row.get("title")
        updated_at = datetime.now(timezone.utc).isoformat()
        state = self.session_manager.peek_session(session_id)
        if state is not None:
            meta = self._session_meta(
                state,
                current_hermes_session_id=current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        else:
            meta = self._provenance_meta(
                session_id,
                current_hermes_session_id or session_id,
                previous_hermes_session_id,
            )
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title if isinstance(title, str) and title.strip() else None,
            updated_at=updated_at,
            field_meta=meta,
        )
        try:
            await self._conn.session_update(session_id=session_id, update=update)
        except Exception:
            logger.debug("Could not send ACP session info update for %s", session_id, exc_info=True)

    def _schedule_usage_update(self, state: SessionState) -> None:
        """Schedule native context indicator refresh after ACP responses."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, self._send_usage_update(state))

    async def _register_session_mcp_servers(
        self,
        state: SessionState,
        mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None,
    ) -> bool:
        """Register ACP-provided MCP servers and retain their safe session config."""
        if not mcp_servers:
            return True

        try:
            from acp_adapter.orchestration import enforce_session_mcp_registration

            config_map: dict[str, dict] = {}
            for server in mcp_servers:
                name = server.name
                if name in config_map:
                    raise RuntimeError(f"duplicate ACP MCP server name: {name}")
                if isinstance(server, McpServerStdio):
                    config = {
                        "command": server.command,
                        "args": list(server.args),
                        "env": {item.name: item.value for item in server.env},
                    }
                else:
                    config = {
                        "url": server.url,
                        "headers": {item.name: item.value for item in server.headers},
                    }
                config_map[name] = enforce_session_mcp_registration(
                    name,
                    config,
                    is_stdio=isinstance(server, McpServerStdio),
                )
        except Exception:
            logger.warning(
                "Session %s: failed to register ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return False

        if not await self._activate_session_mcp_server_configs(state, config_map):
            return False

        # Keep only the post-policy configuration, never the caller-provided
        # ACP objects.  A model switch constructs a replacement agent, so it
        # must replay this exact, already-sanitized session contract before it
        # can become live.
        setattr(state, "_acp_session_mcp_server_configs", deepcopy(config_map))
        return True

    async def _activate_session_mcp_server_configs(
        self,
        state: SessionState,
        config_map: dict[str, dict],
    ) -> bool:
        """Register sanitized MCP config and rebuild ``state.agent``'s tools.

        This is intentionally separate from ACP schema conversion so model
        replacement can reuse only the enforced configuration held by the
        session, rather than reprocessing untrusted client input.
        """
        if not config_map:
            return True

        setattr(
            state.agent,
            "_switchboard_orchestration_mcp_registration_verified",
            False,
        )
        try:
            from tools.mcp_tool import (
                register_mcp_servers,
                registered_mcp_server_matches_config,
            )

            await asyncio.to_thread(register_mcp_servers, config_map)
            if "switchboard_orch" in config_map:
                trusted = await asyncio.to_thread(
                    registered_mcp_server_matches_config,
                    "switchboard_orch",
                    config_map["switchboard_orch"],
                )
                if not trusted:
                    raise RuntimeError(
                        "switchboard_orch is not connected with the trusted "
                        "session launcher configuration"
                    )
                setattr(
                    state.agent,
                    "_switchboard_orchestration_mcp_registration_verified",
                    True,
                )
        except Exception:
            logger.warning(
                "Session %s: failed to activate ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return False

        try:
            from model_tools import get_tool_definitions
            from agent.memory_manager import inject_memory_provider_tools

            enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"],
                mcp_server_names=list(config_map),
            )
            state.agent.enabled_toolsets = enabled_toolsets
            disabled_toolsets = getattr(state.agent, "disabled_toolsets", None)
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            inject_memory_provider_tools(state.agent)
            from acp_adapter.orchestration import without_switchboard_tool_search_bridge

            # Memory providers append directly to agent.tools.  Apply the
            # managed-parent boundary last so no local tool leaks into the
            # provider-facing schema after trusted MCP setup.
            state.agent.tools = without_switchboard_tool_search_bridge(
                state.agent,
                state.agent.tools,
            )
            state.agent.valid_tool_names = {
                tool["function"]["name"] for tool in state.agent.tools or []
            }
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id,
                len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id,
                exc_info=True,
            )
            return False
        return True

    async def _restore_model_switch_mcp_servers(
        self,
        state: SessionState,
        new_agent: Any,
    ) -> bool:
        """Hydrate a replacement agent with the prior session MCP contract."""
        config_map = getattr(state, "_acp_session_mcp_server_configs", None)
        previous_was_verified = bool(
            getattr(
                state.agent,
                "_switchboard_orchestration_mcp_registration_verified",
                False,
            )
        )
        if config_map is None:
            if previous_was_verified:
                logger.error(
                    "Session %s: refusing model switch without the verified "
                    "Switchboard MCP configuration",
                    state.session_id,
                )
                return False
            return True
        if not isinstance(config_map, dict) or any(
            not isinstance(name, str) or not isinstance(config, dict)
            for name, config in config_map.items()
        ):
            logger.error(
                "Session %s: refusing model switch with invalid saved MCP configuration",
                state.session_id,
            )
            return False

        candidate = SimpleNamespace(session_id=state.session_id, agent=new_agent)
        if not await self._activate_session_mcp_server_configs(
            candidate,
            deepcopy(config_map),
        ):
            return False
        if previous_was_verified and not getattr(
            new_agent,
            "_switchboard_orchestration_mcp_registration_verified",
            False,
        ):
            logger.error(
                "Session %s: replacement agent lost verified Switchboard MCP registration",
                state.session_id,
            )
            return False
        if previous_was_verified:
            try:
                from acp_adapter.orchestration import orchestration_meta

                contract = orchestration_meta(new_agent)
            except Exception:
                logger.exception(
                    "Session %s: could not verify replacement Switchboard MCP surface",
                    state.session_id,
                )
                return False
            if not isinstance(contract, dict) or not contract.get("verified"):
                logger.error(
                    "Session %s: replacement agent does not expose the exact "
                    "verified Switchboard MCP surface",
                    state.session_id,
                )
                return False
        return True

    def _schedule_mcp_late_refresh(self, state: SessionState) -> None:
        """Refresh the tool snapshot when background MCP discovery lands after agent build
        (``_make_agent`` only joins ~1.5s). Waits up to 30s off the critical path, then rebuilds
        via ``refresh_agent_mcp_tools`` (same as ``/reload-mcp``).

        Cache safety: only pre-first-turn (nothing cached yet); afterwards the snapshot stays
        frozen and late servers land via the between-turns prologue refresh
        (``agent/turn_context.py``). No-op if discovery finished, join timed out, registry
        unchanged, or session closed."""
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight
        except Exception:
            return
        if not mcp_discovery_in_flight():
            return
        agent, session_id = state.agent, state.session_id

        def _wait_then_refresh() -> None:
            try:
                from hermes_cli.mcp_startup import join_mcp_discovery

                if not join_mcp_discovery(timeout=30.0):
                    return

                # In-memory only: ``get_session()`` would restore from DB and build a new AIAgent.
                with self.session_manager._lock:
                    current = self.session_manager._sessions.get(session_id)
                if current is None or current.agent is not agent:
                    return

                # ``prompt()`` flips ``is_running`` under ``runtime_lock`` before dispatching, so
                # holding it here closes the window where a refresh would swap ``tools=`` mid-turn.
                with current.runtime_lock:
                    if current.is_running:
                        return
                    if any(int(getattr(agent, k, 0) or 0) > 0 for k in ("_user_turn_count", "_api_call_count")):
                        return

                    from tools.mcp_tool_agent import refresh_agent_mcp_tools

                    added = refresh_agent_mcp_tools(agent, quiet_mode=True)
                if added:
                    logger.info(
                        "Session %s: late MCP refresh added %d tools: %s",
                        session_id, len(added), ", ".join(sorted(added)),
                    )
            except Exception:
                logger.debug("Session %s: late MCP refresh failed", session_id, exc_info=True)

        threading.Thread(target=_wait_then_refresh, name=f"acp-mcp-late-refresh-{session_id}", daemon=True).start()

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self, protocol_version: int | None = None, client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None, **kwargs: Any,
    ) -> InitializeResponse:
        auth_methods = build_auth_methods()
        logger.info(
            "Initialize from %s (protocol v%s)", client_info.name if client_info else "unknown",
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(), list=SessionListCapabilities(), resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only acknowledge the method_id advertised in initialize().
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth runs setup out-of-band; succeed only once credentials exist.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay history as user/assistant/thought chunks plus reconstructed tool-call
        start/complete events so the editor shows the transcript, not a clean thread."""
        if not self._conn or not state.history:
            return
        for update in _history_replay_updates(state.history):
            if not await self._send(state.session_id, update, fail_msg="Failed to replay ACP history for session %s"):
                return

    def _guard_owned_session(self, session_id: str, method: str) -> None:
        """Refuse *session_id* unless it belongs to this process's owned set.

        Applies to every protocol handler that takes a client-supplied
        ``session_id`` for an operation other than binding. ``session/new``
        never reaches this (it mints its own id); ``session/load`` and
        ``session/resume`` use :meth:`_guard_first_bind` instead, since
        those two are the legitimate ways an unbound process binds to an
        existing id in the first place.
        """
        owned = self.session_manager.owned_sessions
        if owned.is_owned(session_id):
            return
        primary = owned.primary_id
        logger.warning(
            "cross-session guard: refused %s for session %s; process owns %s",
            method,
            session_id,
            primary,
        )
        raise acp.RequestError(
            _CROSS_SESSION_GUARD_ERROR_CODE,
            f"session {session_id} is not owned by this process",
            {"session_id": session_id, "owned_primary": primary, "method": method},
        )

    def _guard_first_bind(self, session_id: str, method: str) -> None:
        """Bind-on-first-load gate for ``session/load``/``session/resume``.

        The first such call this process handles binds it to *session_id*
        (subject to ``HERMES_EXPECTED_ACP_SESSION_ID`` spawn-time pinning —
        see :meth:`OwnedSessions.check_first_bind`). Once bound, later
        calls to either method fall through to the same ownership check as
        every other protocol handler.
        """
        owned = self.session_manager.owned_sessions
        denial = owned.check_first_bind(session_id)
        if denial is None:
            return
        primary = owned.primary_id
        logger.warning(
            "cross-session guard: refused %s for session %s; process owns %s",
            method,
            session_id,
            primary,
        )
        raise acp.RequestError(
            _CROSS_SESSION_GUARD_ERROR_CODE,
            f"session {session_id} refused: {denial}",
            {"session_id": session_id, "owned_primary": primary, "method": method},
        )

    async def _session_response_fields(self, state: SessionState, replay_verb: str | None = None) -> dict[str, Any]:
        """``models``/``modes``/``field_meta`` for session responses, after an optional history replay;
        schedules command advertisement + usage refresh.

        Per ACP spec, load/resume must stream history via ``session/update`` BEFORE responding
        (Codex/Claude Code/OpenCode/Zed rely on this; deferring via ``call_soon`` broke them).
        Best-effort: a corrupt message must not turn the load into an error."""
        if replay_verb:
            try:
                # Per ACP spec, `session/load` must stream the prior conversation back to the client via
                # `session/update` notifications BEFORE responding, so the client receives the full
                # transcript within the load request's lifetime. Awaiting the replay here matches Codex /
                # Claude Code / OpenCode / Pi and the Zed client (which registers the session-update routing
                # entry before awaiting the loadSession RPC specifically so in-call history replay updates
                # can find the thread). Deferring this via `loop.call_soon` (as we did briefly in May 2026)
                # broke every spec-compliant ACP client that measures notifications synchronously against
                # the load response — see #12285 follow-up.
                await self._replay_session_history(state)
            except Exception:
                logger.warning(
                    f"ACP history replay raised during session/{replay_verb} for %s — "
                    f"{replay_verb} will still succeed, partial transcript may be missing",
                    state.session_id, exc_info=True,
                )
        self._schedule_available_commands_update(state.session_id)
        self._schedule_usage_update(state)
        return {
            "models": self._build_model_state(state),
            "modes": self._session_modes(state),
            "field_meta": self._session_meta(state),
        }

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        self._schedule_available_commands_update(state.session_id)
        self._schedule_usage_update(state)
        return NewSessionResponse(
            session_id=state.session_id,
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            field_meta=self._session_meta(state),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        self._guard_first_bind(session_id, "session/load")
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("Loaded session %s", session_id)
        # Per ACP spec, `session/load` must stream the prior conversation back
        # to the client via `session/update` notifications BEFORE responding,
        # so the client receives the full transcript within the load request's
        # lifetime. Awaiting the replay here matches Codex / Claude Code /
        # OpenCode / Pi and the Zed client (which registers the session-update
        # routing entry before awaiting the loadSession RPC specifically so
        # in-call history replay updates can find the thread). Deferring this
        # via `loop.call_soon` (as we did briefly in May 2026) broke every
        # spec-compliant ACP client that measures notifications synchronously
        # against the load response — see #12285 follow-up.
        try:
            await self._replay_session_history(state)
        except Exception:
            # Replay is best-effort — a corrupted or unexpected message shape
            # must not turn a successful session/load into a JSON-RPC error
            # response. Per-notification failures are already caught inside
            # ``_replay_session_history``; this outer guard covers anything
            # raised by the helpers themselves before reaching ``_send``.
            logger.warning(
                "ACP history replay raised during session/load for %s — "
                "load will still succeed, partial transcript may be missing",
                session_id,
                exc_info=True,
            )
        self._schedule_available_commands_update(session_id)
        self._schedule_usage_update(state)
        return LoadSessionResponse(
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            field_meta=self._session_meta(state),
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        self._guard_first_bind(session_id, "session/resume")
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("Resumed session %s", state.session_id)
        # See `load_session` above for the spec rationale — replay must
        # complete before the response so clients receive the full transcript
        # within the request's lifetime.
        try:
            await self._replay_session_history(state)
        except Exception:
            logger.warning(
                "ACP history replay raised during session/resume for %s — "
                "resume will still succeed, partial transcript may be missing",
                state.session_id,
                exc_info=True,
            )
        self._schedule_available_commands_update(state.session_id)
        self._schedule_usage_update(state)
        return ResumeSessionResponse(
            models=self._build_model_state(state),
            modes=self._session_modes(state),
            field_meta=self._session_meta(state),
        )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._guard_owned_session(session_id, "session/cancel")
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            async with state.turn_terminal_lock:
                with state.runtime_lock:
                    is_running = state.is_running
                    if is_running and state.current_prompt_text:
                        state.interrupted_prompt_text = (
                            state.current_prompt_text
                        )
                    if state.turn_terminal_winner in {
                        "final",
                        "refusal",
                        "provider_error",
                    }:
                        logger.info(
                            "Ignored late cancel for finalized session %s",
                            session_id,
                        )
                        return
                    if is_running:
                        state.turn_terminal_winner = "cancelled"
                    # Publish cancellation AND hard-stop the agent while still
                    # holding runtime_lock: otherwise another prompt can take
                    # the lock in between and mistake this turn for
                    # redirectable work. request_hard_interrupt prefers the
                    # newer hard_interrupt ABI (which also cancels the
                    # compression commit fence) and falls back to
                    # agent.interrupt, so required children are cancelled
                    # either way.
                    state.cancel_event.set()
                    try:
                        if getattr(state, "agent", None):
                            request_hard_interrupt(state.agent)
                    except Exception:
                        logger.debug(
                            "Failed to interrupt ACP session %s",
                            session_id,
                            exc_info=True,
                        )
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        from acp_adapter.orchestration import requested_orchestration_mode

        if requested_orchestration_mode() is not None:
            raise RuntimeError(
                "ACP session/fork is unavailable for a Switchboard-managed "
                "orchestration session"
            )
        self._guard_owned_session(session_id, "session/fork")
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(
            session_id=new_id,
            models=self._build_model_state(state) if state is not None else None,
            modes=self._session_modes(state) if state is not None else None,
            field_meta=self._session_meta(state) if state is not None else None,
        )

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """``cursor`` is a ``session_id`` returned as ``next_cursor``; results resume after it
        (unknown cursor -> empty page, never the full list). Pages cap at the fixed size."""
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        sessions = [
            SessionInfo(
                session_id=s["session_id"], cwd=s["cwd"], title=s.get("title"),
                updated_at=None if s.get("updated_at") is None else str(s["updated_at"]),
            )
            for s in infos[:_LIST_SESSIONS_PAGE_SIZE]
        ]
        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    def _rewrite_prompt_for_interrupt(
        self, state: SessionState, user_text: str, user_content: Any, text_only: bool
    ) -> tuple[str, Any]:
        """Idle ``/steer`` has nothing to inject into (gateway parity): if a prompt was just
        cancelled, replay it with the steer text as explicit correction; otherwise run the steer
        payload as a plain prompt rather than silently queueing it as if ``/queue`` was typed.
        Plain text after a cancel likewise keeps the cancelled request attached ("stop and
        send" clients) so deictic follow-ups have a target."""
        if not (text_only and isinstance(user_content, str)):
            return user_text, user_content

        if user_text.startswith("/steer"):
            split = user_text.split(maxsplit=1)
            steer_text = split[1].strip() if len(split) > 1 else ""
            if not steer_text:
                return user_text, user_content
            idle, interrupted_prompt = _take_interrupted_prompt(state)
            if interrupted_prompt:
                return (_attach_interrupted_prompt(interrupted_prompt, steer_text),) * 2
            return (steer_text, steer_text) if idle else (user_text, user_content)
        if not user_text.startswith("/") and (interrupted_prompt := _take_interrupted_prompt(state)[1]):
            return (_attach_interrupted_prompt(interrupted_prompt, user_text),) * 2
        return user_text, user_content

    def _claim_turn_or_queue(
        self, state: SessionState, session_id: str, user_text: str, user_content: Any, text_only: bool
    ) -> str | None:
        """Mark the session running; if a turn is active, redirect it (text-only, supported
        runtime) or queue it. Returns the client message when absorbed, else None."""
        with state.runtime_lock:
            if not state.is_running:
                state.is_running = True
                state.current_prompt_text = user_text or "[Image attachment]"
                return None
            if text_only and isinstance(user_content, str) and hasattr(state.agent, "redirect") and (
                getattr(state.agent, "_supports_active_turn_redirect", False) is True
            ):
                try:
                    if state.agent.redirect(user_content):
                        return "Redirected the active turn with your correction."
                except Exception:
                    logger.debug("ACP active-turn redirect failed for %s", session_id, exc_info=True)
            state.queued_prompts.append(user_text or "[Image attachment]")
            return f"Queued for the next turn. ({len(state.queued_prompts)} queued)"

    def _run_agent_turn(
        self, *, state: SessionState, session_id: str, user_text: str, user_content: Any, conn: Any,
        loop: asyncio.AbstractEventLoop, approval_cb: Any, edit_approval_requester: Any,
    ) -> dict:
        """Executor-thread body of one turn, run inside ``contextvars.copy_context()`` so
        ContextVar writes are isolated from concurrent sessions.

        Approval routing is thread-local, so it MUST be bound here, not on the loop thread.
        Interactive routing is a ``tools.approval`` contextvar, not ``HERMES_INTERACTIVE`` in
        os.environ, so concurrent workers can't race a global flag onto the non-interactive
        auto-approve path (GHSA-96vc-wcxf-jjff)."""
        agent = state.agent
        with contextlib.ExitStack() as stack:
            # HERMES_SESSION_KEY scopes per-session caches (interactive sudo password) to this
            # session, not the reused thread. ``cwd`` pins what the system prompt reports as the
            # working directory — otherwise it advertises the Hermes workspace while tools are
            # rooted at the client's project and edits land outside it. ``cron_session=""`` masks
            # any leaked process-global HERMES_CRON_SESSION.
            def _session_context() -> Callable[[], None]:
                from gateway.session_context import clear_session_vars, set_session_vars

                tokens = set_session_vars(
                    session_key=session_id, session_id=session_id, cwd=state.cwd, cron_session="",
                )
                return lambda: clear_session_vars(tokens)

            def _approval() -> Callable[[], None]:
                from tools import terminal_tool

                previous = terminal_tool._get_approval_callback()
                terminal_tool.set_approval_callback(approval_cb)
                return lambda: terminal_tool.set_approval_callback(previous)

            def _edit_approval() -> Callable[[], None]:
                from acp_adapter.edit_approval import reset_edit_approval_requester, set_edit_approval_requester

                token = set_edit_approval_requester(edit_approval_requester)
                return lambda: reset_edit_approval_requester(token)

            _bind_guarded(stack, "session context", _session_context)
            if approval_cb:
                _bind_guarded(stack, "approval callback", _approval)
            if edit_approval_requester:
                _bind_guarded(stack, "edit approval requester", _edit_approval)
            stack.callback(reset_hermes_interactive_context, set_hermes_interactive_context(True))
            # Tools tag side-effects with the ACP session (``kanban_create``); save/restore it.
            stack.callback(_restore_env, "HERMES_SESSION_ID", os.environ.get("HERMES_SESSION_ID"))
            os.environ["HERMES_SESSION_ID"] = session_id

            # Auto-titling fires in the turn prologue; push the title now as a session-info update.
            def _notify_title_update(_title: str, _source: str) -> None:
                if conn:
                    loop.call_soon_threadsafe(asyncio.create_task, self._send_session_info_update(session_id))

            agent._on_session_title = _notify_title_update
            try:
                return agent.run_conversation(
                    user_message=user_content, conversation_history=state.history, task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        # ``session/prompt`` cannot establish ownership: doing so would let an
        # arbitrary prompt restore a stranger's durable session. Retain the
        # ACP refusal response for a process that has not bound any session
        # yet, though. Once this process owns a session, every unowned id is
        # still rejected by the cross-session guard below.
        if self.session_manager.owned_sessions.primary_id is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")
        self._guard_owned_session(session_id, "session/prompt")
        try:
            state = self.session_manager.get_session(session_id)
        except UnsafeSessionTranscriptError as exc:
            logger.error("prompt refused: %s", exc)
            if self._conn:
                try:
                    from acp_adapter.tools import _text, make_tool_call_id

                    failure_call_id = make_tool_call_id()
                    await self._conn.session_update(
                        session_id,
                        acp.start_tool_call(
                            failure_call_id,
                            "session transcript safety",
                            kind="execute",
                            raw_input={"status": "unsafe_transcript"},
                        ),
                    )
                    await self._conn.session_update(
                        session_id,
                        acp.update_tool_call(
                            failure_call_id,
                            kind="execute",
                            status="failed",
                            content=[_text(str(exc))],
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Could not emit poisoned transcript refusal for %s",
                        session_id,
                    )
            return PromptResponse(stop_reason="refusal")
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)
        text_only_prompt = all(isinstance(block, TextContentBlock) for block in prompt)
        has_content = bool(user_text) or (
            isinstance(user_content, list) and bool(user_content)
        )
        if not has_content:
            return PromptResponse(stop_reason="end_turn")

        # /steer on an idle session has no in-flight tool call to inject into.
        # Rewrite it so the payload runs as a normal user prompt, matching the
        # gateway's behavior (gateway/run.py ~L4898). Two sub-cases:
        #   1. Zed-interrupt salvage — a prior prompt was cancelled by the
        #      client right before /steer arrived; replay it with the steer
        #      text attached as explicit correction/guidance so the user's
        #      in-flight work isn't lost.
        #   2. Plain idle — no prior work to salvage; just run the steer
        #      payload as a regular prompt. Without this, _cmd_steer would
        #      silently append to state.queued_prompts and respond with
        #      "No active turn — queued for the next turn", which looks like
        #      /queue even though the user never typed /queue.
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/steer"):
            steer_text = user_text.split(maxsplit=1)[1].strip() if len(user_text.split(maxsplit=1)) > 1 else ""
            interrupted_prompt = ""
            rewrite_idle = False
            with state.runtime_lock:
                if not state.is_running and steer_text:
                    if state.interrupted_prompt_text:
                        interrupted_prompt = state.interrupted_prompt_text
                        state.interrupted_prompt_text = ""
                    else:
                        rewrite_idle = True
            if interrupted_prompt:
                user_text = (
                    f"{interrupted_prompt}\n\n"
                    f"User correction/guidance after interrupt: {steer_text}"
                )
                user_content = user_text
            elif rewrite_idle:
                user_text = steer_text
                user_content = steer_text
        elif (
            text_only_prompt
            and isinstance(user_content, str)
            and not user_text.startswith("/")
        ):
            # Some ACP clients implement "stop and send" as two protocol calls:
            # cancel the active prompt, then submit plain correction text. Keep
            # the cancelled request attached so deictic follow-ups ("not that
            # file") still have an explicit target.
            interrupted_prompt = ""
            with state.runtime_lock:
                if not state.is_running and state.interrupted_prompt_text:
                    interrupted_prompt = state.interrupted_prompt_text
                    state.interrupted_prompt_text = ""
            if interrupted_prompt:
                user_text = (
                    f"{interrupted_prompt}\n\n"
                    f"User correction/guidance after interrupt: {user_text}"
                )
                user_content = user_text

        # Intercept slash commands — handle locally without calling the LLM.
        # Slash commands are text-only; if the client included images/resources,
        # send the whole multimodal prompt to the agent instead of treating it as
        # an ACP command.
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if isinstance(response_text, RunPromptAfterCommand):
                # e.g. /goal <text>, /goal resume, /skill <name>: emit the notice, then fall
                # through into the normal turn path below with the sentinel's prompt_text as
                # the user's message — do NOT return end_turn here.
                if self._conn:
                    update = acp.update_agent_message_text(response_text.notice)
                    await self._conn.session_update(session_id, update)
                user_text = response_text.prompt_text
                user_content = response_text.prompt_text
            elif response_text is not None:
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                    await self._send_usage_update(state)
                return PromptResponse(stop_reason="end_turn")

        # If the client sends another regular text prompt while this ACP session
        # is running, route it through the core active-turn redirect. Rich media
        # and older runtimes retain the proven next-turn queue fallback — never
        # race two AIAgent loops against the same state.history. /steer and
        # /queue are handled above and can land immediately.
        #
        # Held under turn_terminal_lock so starting a new turn cannot interleave
        # with cancel()'s terminal-winner arbitration.
        redirected = False
        queued_depth: int | None = None
        async with state.turn_terminal_lock:
            with state.runtime_lock:
                if state.is_running:
                    if (
                        text_only_prompt
                        and isinstance(user_content, str)
                        and getattr(
                            state.agent,
                            "_supports_active_turn_redirect",
                            False,
                        )
                        is True
                        and hasattr(state.agent, "redirect")
                    ):
                        try:
                            redirected = bool(state.agent.redirect(user_content))
                        except Exception:
                            logger.debug(
                                "ACP active-turn redirect failed for %s",
                                session_id,
                                exc_info=True,
                            )
                    if not redirected:
                        queued_text = user_text or "[Image attachment]"
                        state.queued_prompts.append(queued_text)
                        queued_depth = len(state.queued_prompts)
                else:
                    # Fresh turn: clear the previous turn's terminal arbitration
                    # so a prior cancel cannot make this one look pre-cancelled.
                    state.turn_terminal_winner = None
                    if state.cancel_event:
                        state.cancel_event.clear()
                    state.is_running = True
                    state.current_prompt_text = (
                        user_text or "[Image attachment]"
                    )

        if redirected:
            if self._conn:
                update = acp.update_agent_message_text(
                    "Redirected the active turn with your correction."
                )
                await self._conn.session_update(session_id, update)
            return PromptResponse(stop_reason="end_turn")
        # Turn claimed (not queued): keep the gateway stall watchdog fed while
        # silent-but-healthy work runs. Self-terminates when is_running drops,
        # so every return path is covered even without an explicit cancel.
        keepalive_task: asyncio.Task | None = None
        if queued_depth is None:
            keepalive_task = asyncio.create_task(
                self._turn_keepalive_loop(state)
            )

        if queued_depth is not None:
            if self._conn:
                update = acp.update_agent_message_text(
                    f"Queued for the next turn. ({queued_depth} queued)"
                )
                await self._conn.session_update(session_id, update)
            return PromptResponse(stop_reason="end_turn")

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()
        self._ensure_delegation_watcher(loop)

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        tool_call_meta: dict[str, dict[str, Any]] = {}
        delegation_tool_calls: dict[str, str] = {}
        previous_approval_cb = None
        edit_approval_requester = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(
                conn,
                session_id,
                loop,
                tool_call_ids,
                tool_call_meta,
                edit_approval_policy_getter=lambda: self._edit_approval_policy_for_state(state),
            )
            reasoning_cb = make_thinking_cb(conn, session_id, loop)
            base_step_cb = make_step_cb(
                conn, session_id, loop, tool_call_ids, tool_call_meta
            )

            def step_cb(api_call_count: int, prev_tools: Any = None) -> None:
                for tool_info in prev_tools if isinstance(prev_tools, list) else []:
                    if not isinstance(tool_info, dict):
                        continue
                    tool_name = tool_info.get("name") or tool_info.get(
                        "function_name"
                    )
                    result_text = tool_info.get("result") or tool_info.get("output")
                    delegation_id = _async_background_delegation_id(
                        str(result_text) if result_text is not None else None,
                        tool_name,
                    )
                    queue = tool_call_ids.get(tool_name or "")
                    if delegation_id and queue:
                        tool_call_id = (
                            queue
                            if isinstance(queue, str)
                            else next(iter(queue), None)
                        )
                        if tool_call_id:
                            delegation_tool_calls[delegation_id] = tool_call_id
                base_step_cb(api_call_count, prev_tools)
            message_cb = make_message_cb(conn, session_id, loop)

            def stream_delta_cb(text: str) -> None:
                nonlocal streamed_message
                if text:
                    streamed_message = True
                message_cb(text)

            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
            try:
                from acp_adapter.edit_approval import make_acp_edit_approval_requester

                edit_approval_requester = make_acp_edit_approval_requester(
                    conn.request_permission,
                    loop,
                    session_id,
                    auto_approve_getter=lambda: self._edit_approval_policy_for_state(state),
                )
            except Exception:
                logger.debug("Could not create ACP edit approval requester", exc_info=True)
        else:
            tool_progress_cb = None
            reasoning_cb = None
            step_cb = None
            stream_delta_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        # ACP thought panes should not receive Hermes' local kawaii waiting/status
        # updates. Route provider/model reasoning deltas instead; if the provider
        # emits no reasoning, Zed should not get a fake "thinking" accordion.
        agent.thinking_callback = None
        agent.reasoning_callback = reasoning_cb
        agent.step_callback = step_cb
        agent.stream_delta_callback = stream_delta_cb

        # Approval callback is per-thread (thread-local, GHSA-qg5c-hvr5-hjgr).
        # Set it INSIDE _run_agent so the TLS write happens in the executor
        # thread — setting it here would write to the event-loop thread's TLS,
        # not the executor's. Interactive routing uses a contextvar in
        # tools.approval (set_hermes_interactive_context) rather than
        # os.environ["HERMES_INTERACTIVE"], so concurrent executor workers can't
        # race on a process-global flag — one session's restore can't drop
        # another onto the non-interactive auto-approve path mid-run
        # (GHSA-96vc-wcxf-jjff). The contextvar write is isolated by the
        # contextvars.copy_context() wrapper around the executor call below.
        # ACP's conn.request_permission maps cleanly to the interactive
        # callback shape — not the gateway-queue HERMES_EXEC_ASK path,
        # which requires a notify_cb registered in _gateway_notify_cbs.
        previous_approval_cb = None
        interactive_token = None
        edit_approval_token = None
        previous_session_id = None

        def _run_agent(
            run_user_content=user_content,
            persist_user_message=user_text or "[Image attachment]",
        ) -> dict:
            nonlocal previous_approval_cb, interactive_token, edit_approval_token, previous_session_id
            # Bind HERMES_SESSION_KEY for this session so per-session caches
            # (e.g. the interactive sudo password cache in tools.terminal_tool)
            # scope to the ACP session rather than leaking across sessions
            # that land on the same reused executor thread. This call runs
            # inside a contextvars.copy_context() below, so the ContextVar
            # write is isolated from other concurrent ACP sessions.
            try:
                from gateway.session_context import (
                    clear_session_vars,
                    set_session_vars,
                )
                # ``cwd`` pins the logical working directory for this context,
                # which is what the system prompt's "Current working directory"
                # line reports (agent/prompt_builder.py -> resolve_agent_cwd).
                # Without it the prompt advertises the global Hermes workspace
                # while the tools are rooted at the client's project, so the
                # model emits absolute paths under ~/.hermes/workspace and the
                # edit silently lands outside the editor's workspace.
                # cron_session="" explicitly marks this as a non-cron context,
                # masking any leaked process-global HERMES_CRON_SESSION (#37968).
                session_tokens = set_session_vars(
                    session_key=session_id, session_id=session_id, cwd=state.cwd,
                    cron_session="",
                )
            except Exception:
                session_tokens = None
                clear_session_vars = None  # type: ignore[assignment]
                logger.debug("Could not set ACP session context", exc_info=True)
            if approval_cb:
                try:
                    from tools import terminal_tool as _terminal_tool
                    previous_approval_cb = _terminal_tool._get_approval_callback()
                    _terminal_tool.set_approval_callback(approval_cb)
                except Exception:
                    logger.debug("Could not set ACP approval callback", exc_info=True)
            if edit_approval_requester:
                try:
                    from acp_adapter.edit_approval import set_edit_approval_requester

                    edit_approval_token = set_edit_approval_requester(edit_approval_requester)
                except Exception:
                    logger.debug("Could not set ACP edit approval requester", exc_info=True)
            # Signal to tools.approval that we have an interactive callback
            # and the non-interactive auto-approve path must not fire. Uses a
            # contextvar (not os.environ) so concurrent executor workers don't
            # race on the flag (GHSA-96vc-wcxf-jjff).
            interactive_token = set_hermes_interactive_context(True)
            # Propagate the originating ACP session id to tools that want to
            # tag side-effects with it (e.g. ``kanban_create`` stamps it on
            # the new task so clients can render a per-session board). Save
            # and restore around the agent call so a re-used executor thread
            # never leaks one session's id into the next session's tools.
            previous_session_id = os.environ.get("HERMES_SESSION_ID")
            os.environ["HERMES_SESSION_ID"] = session_id
            # Auto-titling fires inside the turn prologue now; give the agent
            # this session's notifier so a new title reaches the client as a
            # session-info update instead of waiting for the next one.
            def _notify_title_update(_title: str, _source: str) -> None:
                if conn:
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        self._send_session_info_update(session_id),
                    )

            agent._on_session_title = _notify_title_update
            try:
                result = agent.run_conversation(
                    user_message=run_user_content,
                    conversation_history=state.history,
                    task_id=session_id,
                    persist_user_message=persist_user_message,
                )
                try:
                    required_pending = bool(
                        getattr(agent, "_required_delegation_launching", False)
                        or agent._has_unconsumed_required_delegations()
                    )
                except Exception:
                    required_pending = True
                if required_pending:
                    # Last-line ACP integrity boundary for direct-return/fatal
                    # paths inside the conversation loop. They must never turn
                    # an unobserved child into a normal assistant answer.
                    try:
                        from tools.async_delegation import (
                            stop_required_for_agent,
                        )

                        stop_required_for_agent(
                            agent,
                            reason=(
                                "ACP parent exited before required child "
                                "observation completed"
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "ACP required delegation boundary cleanup failed"
                        )
                    try:
                        agent._finish_acp_provisional_stream(discard=True)
                    except Exception:
                        logger.debug(
                            "ACP required provisional cleanup failed",
                            exc_info=True,
                        )
                    safe_messages = _sanitize_failed_turn_history(
                        result.get("messages") or state.history,
                        baseline_count=turn_history_baseline_count,
                    )
                    history_rewrite_succeeded = _rewrite_agent_active_history(
                        agent,
                        safe_messages,
                        state,
                        self.session_manager,
                    )
                    return {
                        "final_response": None,
                        "messages": safe_messages,
                        "completed": False,
                        "failed": True,
                        "error": (
                            "required_delegation_observation_failed"
                            if history_rewrite_succeeded
                            else "required_delegation_observation_persistence_failed"
                        ),
                        "history_rewrite_succeeded": history_rewrite_succeeded,
                    }
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                try:
                    required_pending = bool(
                        getattr(agent, "_required_delegation_launching", False)
                        or agent._has_unconsumed_required_delegations()
                    )
                except Exception:
                    # If the integrity check itself fails at the ACP boundary,
                    # never turn the caught exception into authoritative prose.
                    required_pending = True
                if required_pending:
                    # Same last-line ACP integrity boundary as the success-path
                    # branch above: an exception unwinding out of
                    # run_conversation must not leave an owned required record
                    # (and its child work) resident with no future turn to
                    # terminalize it.
                    try:
                        from tools.async_delegation import (
                            stop_required_for_agent,
                        )

                        stop_required_for_agent(
                            agent,
                            reason=(
                                "ACP parent raised before required child "
                                "observation completed"
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "ACP required delegation boundary cleanup failed"
                        )
                    safe_messages = _sanitize_failed_turn_history(
                        state.history,
                        baseline_count=turn_history_baseline_count,
                    )
                    history_rewrite_succeeded = _rewrite_agent_active_history(
                        agent,
                        safe_messages,
                        state,
                        self.session_manager,
                    )
                    return {
                        "final_response": "",
                        "messages": safe_messages,
                        "completed": False,
                        "failed": True,
                        "error": (
                            "required_delegation_observation_failed"
                            if history_rewrite_succeeded
                            else "required_delegation_observation_persistence_failed"
                        ),
                        "history_rewrite_succeeded": history_rewrite_succeeded,
                    }
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                # Restore the interactive contextvar for this context.
                if interactive_token is not None:
                    reset_hermes_interactive_context(interactive_token)
                # Restore HERMES_SESSION_ID symmetrically.
                if previous_session_id is None:
                    os.environ.pop("HERMES_SESSION_ID", None)
                else:
                    os.environ["HERMES_SESSION_ID"] = previous_session_id
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)
                if edit_approval_token is not None:
                    try:
                        from acp_adapter.edit_approval import reset_edit_approval_requester

                        reset_edit_approval_requester(edit_approval_token)
                    except Exception:
                        logger.debug("Could not restore ACP edit approval requester", exc_info=True)
                if session_tokens is not None and clear_session_vars is not None:
                    try:
                        clear_session_vars(session_tokens)
                    except Exception:
                        logger.debug("Could not clear ACP session context", exc_info=True)

        while True:
            streamed_message = False
            turn_history_baseline_count = len(state.history)
            should_continue_goal = False
            next_turn_text: str | None = None

            try:
                # Snapshot the internal Hermes DB session id before the turn so we
                # can detect a compression-driven session rotation afterwards. The
                # ACP `session_id` stays the stable client handle; agent.session_id
                # is the live internal head that compression may rotate.
                pre_turn_hermes_id = getattr(state.agent, "session_id", None)
                # Wrap the executor call in a fresh copy of the current context so
                # concurrent ACP sessions on the shared ThreadPoolExecutor don't
                # stomp on each other's ContextVar writes (HERMES_SESSION_KEY in
                # particular — used by the interactive sudo password cache scope).
                ctx = contextvars.copy_context()
                turn_start_ts = time.time()
                result = await loop.run_in_executor(
                    _executor, ctx.run, _run_agent, user_content, (user_text or "[Image attachment]"),
                )
            except Exception:
                logger.exception("Executor error for session %s", session_id)
                async with state.turn_terminal_lock:
                    executor_cancelled = bool(
                        state.turn_terminal_winner == "cancelled"
                        or (
                            state.cancel_event
                            and state.cancel_event.is_set()
                        )
                    )
                    state.turn_terminal_winner = (
                        "cancelled" if executor_cancelled else "refusal"
                    )
                with state.runtime_lock:
                    state.is_running = False
                    state.current_prompt_text = ""
                if conn:
                    flush_open_tool_calls(conn, session_id, loop, tool_call_ids, tool_call_meta)
                    for update in flush_async_background_dispatches(
                        delegation_tool_calls, set(delegation_tool_calls)
                    ):
                        await conn.session_update(session_id, update)
                # HOLE 2 (Phase 0 / stop-p0-brief.md): this path must drain
                # queued_prompts exactly like every other return path below, so a
                # prompt the user typed while the initial executor dispatch was
                # crashing doesn't sit stuck forever.
                await self._drain_queued_prompts(state, session_id, conn)
                return PromptResponse(
                    stop_reason="cancelled" if executor_cancelled else "end_turn"
                )

            if result.get("messages"):
                state.history = result["messages"]

            joined_completed_ids: set[str] = set()
            # Everything from the same-turn delegation barrier through delivering
            # the final response is wrapped in this try/finally so a turn that
            # cooperatively returns from the executor ALWAYS frees the session
            # (`is_running=False`) and drains anything queued while it ran — on
            # normal, cancelled, or errored exit. No path below may leave
            # `is_running` True (Phase 0 / stop-p0-brief.md P0.1).
            # Computed before the barrier (Phase 0 / stop-p0-brief.md P0.2): a
            # turn that was cancelled must skip the join barrier entirely rather
            # than re-running the agent with a "consolidate results" prompt the
            # user never asked for.
            cancelled_before_barrier = bool(
                state.cancel_event and state.cancel_event.is_set()
            )
            try:
                if cancelled_before_barrier:
                    try:
                        from tools.async_delegation import interrupt_for_session

                        interrupt_for_session(
                            session_key=session_id, reason="user cancelled"
                        )
                    except Exception:
                        logger.exception(
                            "Failed to interrupt async delegations for cancelled "
                            "session %s",
                            session_id,
                        )
                else:
                    try:
                        from tools.async_delegation import join, running_for_session
                        from tools.delegate_tool import _load_config
                        from tools.process_registry import (
                            format_process_notification,
                            process_registry,
                        )
                        from utils import is_truthy_value

                        delegation_config = _load_config()
                        join_enabled = is_truthy_value(
                            delegation_config.get("acp_join_same_turn"), default=True
                        )
                        try:
                            max_join_rounds = max(
                                0, int(delegation_config.get("acp_join_max_rounds", 3))
                            )
                        except (TypeError, ValueError):
                            max_join_rounds = 3
                        try:
                            join_timeout = max(
                                0.0,
                                float(
                                    delegation_config.get(
                                        "acp_join_timeout_seconds", 180
                                    )
                                ),
                            )
                        except (TypeError, ValueError):
                            join_timeout = 180.0

                        def _cancelled() -> bool:
                            return bool(
                                state.cancel_event and state.cancel_event.is_set()
                            )

                        def _interrupt_cancelled_subagents(where: str) -> None:
                            try:
                                from tools.async_delegation import (
                                    interrupt_for_session,
                                )

                                interrupt_for_session(
                                    session_key=session_id, reason="user cancelled"
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to interrupt async delegations for "
                                    "session %s cancelled %s",
                                    session_id,
                                    where,
                                )

                        pending = (
                            running_for_session(session_id, turn_start_ts)
                            if join_enabled
                            else []
                        )
                        pending_note_added = False
                        for _round in range(max_join_rounds):
                            if not pending:
                                # Nothing left to join. Still interrupt if STOP
                                # landed exactly as the last round's pending list
                                # emptied out, so a race here can't leave a
                                # subagent running unattended past cancellation
                                # (Phase 0 / stop-p0-brief.md HOLE 1).
                                if _cancelled():
                                    _interrupt_cancelled_subagents("with nothing pending")
                                break
                            # Re-check at the TOP of each round: STOP can land while
                            # we're waiting on join() below, or during the
                            # continuation re-run's executor call. Abort the
                            # barrier loop rather than keep consolidating a turn
                            # the user already cancelled.
                            if _cancelled():
                                _interrupt_cancelled_subagents("mid-join")
                                break
                            delegation_ids = [
                                str(record.get("delegation_id") or "")
                                for record in pending
                                if record.get("delegation_id")
                            ]
                            joined = await loop.run_in_executor(
                                _executor, join, delegation_ids, join_timeout
                            )
                            # Re-check IMMEDIATELY after join() returns: STOP can
                            # land while this coroutine was blocked waiting on the
                            # executor, before any of the completed delegations'
                            # results are consolidated into a continuation turn
                            # (Phase 0 / stop-p0-brief.md HOLE 1, checkpoint a).
                            if _cancelled():
                                _interrupt_cancelled_subagents("right after join() returned")
                                break
                            joined_completed_ids.update(
                                str(delegation_id)
                                for delegation_id in joined.get("completed") or []
                            )
                            completed_events = self._drain_session_delegation_completions(
                                process_registry,
                                format_process_notification,
                                session_id,
                                state,
                            )
                            for event in completed_events:
                                delegation_id = str(event.get("delegation_id") or "")
                                tool_call_id = delegation_tool_calls.pop(
                                    delegation_id, None
                                )
                                if conn and tool_call_id:
                                    formatted_result = format_process_notification(event) or str(
                                        event.get("summary") or event.get("error") or ""
                                    )
                                    await conn.session_update(
                                        session_id,
                                        build_async_background_completion(
                                            tool_call_id, event, formatted_result
                                        ),
                                    )
                            if joined.get("pending"):
                                state.history.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            "Background subagent(s) still running; results "
                                            "will arrive shortly."
                                        ),
                                    }
                                )
                                pending_note_added = True
                                break
                            if not completed_events:
                                break

                            # Re-check IMMEDIATELY BEFORE dispatching the
                            # continuation: STOP can land during the completion
                            # drain/notification work above, between join()
                            # returning and this dispatch (Phase 0 /
                            # stop-p0-brief.md HOLE 1, checkpoint b).
                            if _cancelled():
                                _interrupt_cancelled_subagents(
                                    "right before the continuation dispatch"
                                )
                                break

                            continuation = (
                                "Your background subagent(s) have completed; their results "
                                "are above. Incorporate them and give your consolidated "
                                "final answer."
                            )
                            continuation_ctx = contextvars.copy_context()
                            result = await loop.run_in_executor(
                                _executor,
                                continuation_ctx.run,
                                _run_agent,
                                continuation,
                                continuation,
                            )
                            if result.get("messages"):
                                state.history = result["messages"]
                            pending = running_for_session(session_id, turn_start_ts)

                        if pending and not pending_note_added:
                            state.history.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Background subagent(s) still running; results will "
                                        "arrive shortly."
                                    ),
                                }
                            )
                    except Exception:
                        logger.exception(
                            "ACP same-turn delegation join failed for session %s", session_id
                        )

                missing_result_ids = {
                    delegation_id
                    for delegation_id in joined_completed_ids
                    if delegation_id in delegation_tool_calls
                }

                # The turn is over: close any tool calls whose completion never made it
                # through the name-keyed FIFO pairing (long-turn steering/compression
                # drift), so clients never keep stuck in-progress items past end_turn.
                if conn:
                    flush_open_tool_calls(conn, session_id, loop, tool_call_ids, tool_call_meta)
                    for update in flush_async_background_dispatches(
                        delegation_tool_calls, missing_result_ids
                    ):
                        await conn.session_update(session_id, update)

                # Detect a compression-driven internal session rotation. If the agent's
                # DB head moved during the turn, emit a session_info_update carrying
                # _meta.hermes.sessionProvenance so ACP clients can render the boundary
                # and keep old/new ids in lineage. The ACP session_id is unchanged.
                post_turn_hermes_id = getattr(state.agent, "session_id", None)
                if (
                    conn
                    and post_turn_hermes_id
                    and pre_turn_hermes_id
                    and post_turn_hermes_id != pre_turn_hermes_id
                ):
                    try:
                        await self._send_session_info_update(
                            session_id,
                            current_hermes_session_id=post_turn_hermes_id,
                            previous_hermes_session_id=pre_turn_hermes_id,
                        )
                    except Exception:
                        logger.debug(
                            "Could not emit ACP provenance update after rotation for %s",
                            session_id,
                            exc_info=True,
                        )

                final_response = str(result.get("final_response") or "")
                cancelled = False
                interrupted = bool(result.get("interrupted"))
                required_failure_code = result.get("error")
                required_observation_failed = required_failure_code in {
                    "required_delegation_observation_failed",
                    "required_delegation_observation_persistence_failed",
                }
                # Every "give up" exit in conversation_loop.py's retry/fallback
                # ladder (max API retries, billing wall, thinking timeout, ...)
                # sets failed=True and stuffs a human-readable explanation into
                # final_response. Delivering that as ordinary assistant prose
                # makes a provider outage indistinguishable from a real answer;
                # route it through a genuine JSON-RPC error instead (raised at
                # the end of this try body) so the gateway's existing error feed
                # path renders it distinctly. required_observation_failed keeps
                # its own dedicated async-delegation handling.
                provider_call_failed = (
                    bool(result.get("failed")) and not required_observation_failed
                )
                history_rewrite_failed = (
                    required_failure_code
                    == "required_delegation_observation_persistence_failed"
                )
                if required_observation_failed:
                    # Treat the error flag as authoritative even if a malformed
                    # internal result also carried candidate prose.
                    result["final_response"] = ""
                    final_response = ""
                    state.history = _sanitize_failed_turn_history(
                        result.get("messages") or state.history,
                        baseline_count=turn_history_baseline_count,
                    )
                    result["messages"] = state.history
                    if "history_rewrite_succeeded" not in result:
                        history_rewrite_succeeded = await asyncio.to_thread(
                            _rewrite_agent_active_history,
                            state.agent,
                            state.history,
                            state,
                            self.session_manager,
                        )
                        history_rewrite_failed = not history_rewrite_succeeded
                    poison_marker_failed = (
                        state.transcript_correction_poisoned
                        and state.transcript_correction_poison_persisted is False
                    )
                    if poison_marker_failed:
                        required_error_text = (
                            "Hermes could not durably correct the cancelled turn "
                            "or record its transcript safety marker. No final "
                            "answer was produced; do not resume this session "
                            "because stale candidate replay cannot be ruled out."
                        )
                    elif history_rewrite_failed:
                        required_error_text = (
                            "Hermes could not durably correct the cancelled turn. "
                            "No final answer was produced; this session is blocked "
                            "from resume until its transcript is repaired."
                        )
                    else:
                        required_error_text = (
                            "Hermes could not safely persist the required child "
                            "result. No final answer was produced; the child work "
                            "was cancelled to keep this turn consistent."
                        )
                    logger.error(
                        "ACP required-delegation observation failed for session %s",
                        session_id,
                    )
                    if conn:
                        try:
                            from acp_adapter.tools import (
                                _text as _tool_text,
                                make_tool_call_id,
                            )

                            failure_call_id = make_tool_call_id()
                            await conn.session_update(
                                session_id,
                                acp.start_tool_call(
                                    failure_call_id,
                                    "required delegation observation",
                                    kind="execute",
                                    raw_input={
                                        "tool": "delegation_wait",
                                        "arguments": {
                                            "status": (
                                                "persistence_failed"
                                                if history_rewrite_failed
                                                else "observation_failed"
                                            ),
                                        },
                                    },
                                ),
                            )
                            await conn.session_update(
                                session_id,
                                acp.update_tool_call(
                                    failure_call_id,
                                    kind="execute",
                                    status="failed",
                                    content=[_tool_text(required_error_text)],
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "Could not emit ACP required-delegation failure "
                                "for session %s",
                                session_id,
                            )
                    # Emit/log the structured failure before abandoning the old
                    # owner. The next user turn must not inherit or strand a
                    # process-local record tied to this turn id.
                    try:
                        from tools.async_delegation import stop_required_for_agent

                        stop_required_for_agent(
                            state.agent,
                            reason=(
                                "required child result could not be persisted "
                                "atomically"
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "Could not abandon failed required delegation for "
                            "session %s",
                            session_id,
                        )
                    state.agent._required_delegation_launching = False
                    state.agent._required_observation_failed = False
                    try:
                        state.agent._finish_acp_provisional_stream(discard=True)
                    except Exception:
                        logger.debug(
                            "Required failure provisional cleanup failed",
                            exc_info=True,
                        )
                # Hermes' local "waiting for model response" interrupt status is metadata,
                # not assistant prose — clients get cancellation from stop_reason instead.
                from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

                # Final delivery, durable history, and STOP share one per-session
                # terminal claim. Once final owns this lock, cancel() waits and
                # observes the committed winner instead of changing an already
                # delivered answer into a cancelled turn.
                async with state.turn_terminal_lock:
                    event_cancelled = bool(
                        state.cancel_event
                        and state.cancel_event.is_set()
                    )
                    if (
                        state.turn_terminal_winner == "cancelled"
                        or (
                            state.turn_terminal_winner != "final"
                            and event_cancelled
                        )
                    ):
                        state.turn_terminal_winner = "cancelled"
                        cancelled = True
                        interrupted = True
                        result["final_response"] = ""
                        final_response = ""
                        state.history = _sanitize_failed_turn_history(
                            result.get("messages") or state.history,
                            baseline_count=turn_history_baseline_count,
                        )
                        result["messages"] = state.history
                        history_rewrite_succeeded = await asyncio.to_thread(
                            _rewrite_agent_active_history,
                            state.agent,
                            state.history,
                            state,
                            self.session_manager,
                        )
                        history_rewrite_failed = not history_rewrite_succeeded
                        try:
                            state.agent._finish_acp_provisional_stream(
                                discard=True
                            )
                        except Exception:
                            logger.debug(
                                "Cancelled ACP provisional cleanup failed",
                                exc_info=True,
                            )
                    elif required_observation_failed:
                        state.turn_terminal_winner = "refusal"
                    elif provider_call_failed:
                        state.turn_terminal_winner = "provider_error"
                        # Same failed-turn history semantics as a cancelled turn:
                        # strip the fabricated failure-text assistant candidate so
                        # it neither reaches the model as genuine prior context on
                        # the next turn nor replays as an ordinary answer after a
                        # session reload. final_response is intentionally kept —
                        # it becomes the JSON-RPC error message below.
                        state.history = _sanitize_failed_turn_history(
                            result.get("messages") or state.history,
                            baseline_count=turn_history_baseline_count,
                        )
                        result["messages"] = state.history
                        history_rewrite_succeeded = await asyncio.to_thread(
                            _rewrite_agent_active_history,
                            state.agent,
                            state.history,
                            state,
                            self.session_manager,
                        )
                        history_rewrite_failed = not history_rewrite_succeeded
                        try:
                            state.agent._finish_acp_provisional_stream(
                                discard=True
                            )
                        except Exception:
                            logger.debug(
                                "provider_error provisional cleanup failed",
                                exc_info=True,
                            )
                    else:
                        state.turn_terminal_winner = "final"

                    suppress_interrupt_response = (
                        interrupted
                        and final_response.startswith(
                            INTERRUPT_WAITING_FOR_MODEL_PREFIX
                        )
                    )
                    if result.get("messages"):
                        # Persist only after the terminal winner sanitizes the
                        # turn, and keep the commit serialized with final delivery.
                        self.session_manager.save_session(session_id)
                    if (
                        state.turn_terminal_winner == "final"
                        and final_response
                        and conn
                        and not suppress_interrupt_response
                        and (
                            not streamed_message
                            or result.get("response_transformed")
                        )
                    ):
                        # Deliver the final response when streaming did not already
                        # send it, or when a plugin transformed it afterwards.
                        update = acp.update_agent_message_text(final_response)
                        await conn.session_update(session_id, update)

                    terminal_winner = state.turn_terminal_winner

                if (
                    history_rewrite_failed
                    and not required_observation_failed
                    and conn
                ):
                    try:
                        from acp_adapter.tools import (
                            _text as _tool_text,
                            make_tool_call_id,
                        )

                        failure_call_id = make_tool_call_id()
                        if (
                            state.transcript_correction_poisoned
                            and state.transcript_correction_poison_persisted
                            is False
                        ):
                            failure_text = (
                                "Hermes cancelled this turn but could not durably "
                                "remove the rejected assistant candidate or record "
                                "its transcript safety marker. Do not resume this "
                                "session because stale replay cannot be ruled out."
                            )
                        else:
                            failure_text = (
                                "Hermes cancelled this turn but could not durably "
                                "remove the rejected assistant candidate. No final "
                                "answer was delivered; this session is blocked "
                                "from resume until its transcript is repaired."
                            )
                        await conn.session_update(
                            session_id,
                            acp.start_tool_call(
                                failure_call_id,
                                "turn history correction",
                                kind="execute",
                                raw_input={
                                    "status": "persistence_failed",
                                },
                            ),
                        )
                        await conn.session_update(
                            session_id,
                            acp.update_tool_call(
                                failure_call_id,
                                kind="execute",
                                status="failed",
                                content=[_tool_text(failure_text)],
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "Could not emit ACP transcript-correction failure "
                            "for session %s",
                            session_id,
                        )

                if (
                    terminal_winner == "final"
                    and final_response
                    and not suppress_interrupt_response
                ):
                    try:
                        from agent.title_generator import maybe_auto_title

                        def _notify_title_update(_title: str) -> None:
                            if conn:
                                loop.call_soon_threadsafe(
                                    asyncio.create_task,
                                    self._send_session_info_update(session_id),
                                )

                        # Snapshot the runtime identity; the validator lets the
                        # background titler skip its LLM call if the session's model
                        # changed before it fires (#19027).
                        _title_model = getattr(state.agent, "model", None)
                        _title_provider = getattr(state.agent, "provider", None)
                        maybe_auto_title(
                            self.session_manager._get_db(),
                            session_id,
                            user_text,
                            final_response,
                            state.history,
                            main_runtime={
                                "model": getattr(state.agent, "model", None),
                                "provider": getattr(state.agent, "provider", None),
                                "base_url": getattr(state.agent, "base_url", None),
                                "api_key": getattr(state.agent, "api_key", None),
                                "api_mode": getattr(state.agent, "api_mode", None),
                            },
                            runtime_validator=lambda: (
                                getattr(state.agent, "model", None) == _title_model
                                and getattr(state.agent, "provider", None) == _title_provider
                            ),
                            title_callback=_notify_title_update,
                        )
                    except Exception:
                        logger.debug("Failed to auto-title ACP session %s", session_id, exc_info=True)

                if terminal_winner == "provider_error":
                    # End this session/prompt request with a JSON-RPC error so
                    # clients render the failure distinctly instead of as chat.
                    # The acp dispatcher converts this into an error response for
                    # this request id without crashing the process or connection,
                    # and the finally: below still resets is_running and drains
                    # queued prompts, so the session is never left wedged.
                    # Raised even when partial text already streamed (mirrors
                    # cancelled-turn semantics: visible partial text + a terminal
                    # marker); gating on streamed_message would resurrect the
                    # silent-failure bug for every mid-stream provider drop.
                    # The usage gauge is refreshed first because the normal
                    # post-try update is skipped when this raise unwinds.
                    await self._send_usage_update(state)
                    raise acp.RequestError(
                        -32001,
                        final_response
                        or "Hermes turn failed: the provider call did not succeed.",
                        {
                            "failureReason": result.get("failure_reason"),
                            "billingBlock": result.get("billing_block"),
                        },
                    )
                async def _pause_goal_on_interrupt() -> None:
                    # CLI parity (cli_loops_mixin._maybe_continue_goal_after_turn):
                    # an interrupted/cancelled turn auto-pauses the goal instead of
                    # judging (or acting on) partial output -- the judge would
                    # almost always say "continue" and re-queue exactly what was
                    # just interrupted.
                    goal_mgr = _get_goal_manager(state)
                    if goal_mgr.is_active():
                        goal_mgr.pause(reason="user-interrupted (Ctrl+C)")
                        if conn:
                            await conn.session_update(
                                session_id,
                                acp.update_agent_message_text(
                                    "⏸ Goal paused — turn was interrupted. "
                                    "Use /goal resume to continue, or /goal clear to stop."
                                ),
                            )

                # Phase 3b: after a normal turn completion, let an active goal judge
                # whether to keep going. This reuses GoalManager.evaluate_after_turn
                # (hermes_cli/goals.py) exactly as the CLI/gateway do -- same budget
                # accounting, gates, judge call, and notice wording -- so the loop
                # itself is just: stream the decision's message, then either run
                # another turn with its continuation_prompt or stop. Stop conditions
                # are checked before AND after the (possibly slow) judge call, mirroring
                # cli_loops_mixin._maybe_continue_goal_after_turn: a real user prompt
                # already queued always wins over continuing, and a cancel that races
                # in while the judge is deliberating must not start another turn.
                if (
                    terminal_winner == "final"
                    and not cancelled
                    and not required_observation_failed
                    and final_response
                    and not suppress_interrupt_response
                ):
                    if state.cancel_event and state.cancel_event.is_set():
                        cancelled = True
                        await _pause_goal_on_interrupt()
                    else:
                        goal_mgr = _get_goal_manager(state)
                        with state.runtime_lock:
                            goal_queue_waiting = bool(state.queued_prompts)
                        if goal_mgr.is_active() and not goal_queue_waiting:
                            try:
                                from hermes_cli.goals import gather_background_processes

                                goal_bg_procs = gather_background_processes()
                            except Exception:
                                goal_bg_procs = None
                            # This turn's completion already committed
                            # state.turn_terminal_winner = "final" above. Clear it
                            # now, before the (possibly slow) judge call below, so
                            # a real session/cancel racing in while the judge
                            # deliberates isn't dropped by cancel()'s
                            # finalized-session guard -- mirrors the fresh-turn
                            # reset used when a new prompt() call claims an idle
                            # session.
                            async with state.turn_terminal_lock:
                                state.turn_terminal_winner = None
                            # judge_goal is a blocking HTTP call to the configured
                            # auxiliary.goal_judge model (seconds, sometimes tens of
                            # seconds) -- run it off the event loop like the main
                            # turn above, not inline.
                            try:
                                goal_decision = await loop.run_in_executor(
                                    _executor,
                                    lambda: goal_mgr.evaluate_after_turn(
                                        final_response,
                                        user_initiated=True,
                                        background_processes=goal_bg_procs,
                                    ),
                                )
                            except Exception as exc:
                                # A judge/evaluate crash (bad judge model config,
                                # transport bug, ...) must not take the whole ACP
                                # turn down with it -- pause the goal so the user
                                # sees why and can /goal resume, same as any other
                                # judge failure mode above.
                                logger.warning(
                                    "Goal evaluate_after_turn raised for session %s",
                                    session_id,
                                    exc_info=True,
                                )
                                goal_mgr.pause("judge-error")
                                if conn:
                                    await conn.session_update(
                                        session_id,
                                        acp.update_agent_message_text(
                                            f"⏸ Goal paused — judge error: {exc}"
                                        ),
                                    )
                            else:
                                if state.cancel_event and state.cancel_event.is_set():
                                    cancelled = True
                                    await _pause_goal_on_interrupt()
                                else:
                                    if goal_decision.get("message") and conn:
                                        await conn.session_update(
                                            session_id,
                                            acp.update_agent_message_text(goal_decision["message"]),
                                        )
                                    with state.runtime_lock:
                                        goal_queue_waiting = bool(state.queued_prompts)
                                    if (
                                        goal_decision.get("should_continue")
                                        and goal_decision.get("continuation_prompt")
                                        and not goal_queue_waiting
                                    ):
                                        should_continue_goal = True
                                        next_turn_text = goal_decision["continuation_prompt"]
                elif (
                    (cancelled or suppress_interrupt_response)
                    and not required_observation_failed
                ):
                    await _pause_goal_on_interrupt()
            finally:
                # Mark this turn idle before draining queued work so recursive prompt()
                # calls can acquire the session. Queued turns are intentionally run as
                # normal follow-up user prompts, preserving role alternation and history.
                # This reset + the drain below MUST run no matter how the block above
                # exits (success, caught error, or an uncaught exception) so a turn
                # that cooperatively returned from the executor never leaves the
                # session wedged with is_running=True and an undrained queue.
                #
                # Skipped when a goal continuation is about to run another turn in
                # this same prompt() call (`should_continue_goal`): is_running must
                # stay True and the keepalive loop must keep feeding the stall
                # watchdog across iterations (Phase 3b contract). The queued-prompt
                # check above already guarantees `should_continue_goal` is False
                # whenever a real user prompt is waiting, so it still gets drained
                # promptly once the loop does stop.
                if not should_continue_goal:
                    with state.runtime_lock:
                        state.is_running = False
                        state.current_prompt_text = ""

                    if keepalive_task is not None:
                        keepalive_task.cancel()

                    await self._drain_queued_prompts(state, session_id, conn)

            if should_continue_goal:
                user_text = next_turn_text
                user_content = next_turn_text
                continue
            break

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        await self._send_usage_update(state)

        cancelled = bool(
            cancelled or terminal_winner == "cancelled"
        )
        stop_reason = (
            "cancelled"
            if cancelled
            else "refusal"
            if required_observation_failed
            else "end_turn"
        )
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Session settings (ACP protocol methods) -----------------------------

    def _cmd_version(self, args: str, state: SessionState) -> str:
        """Show the Hermes version together with the active Git identity."""
        try:
            from hermes_cli.banner import get_git_build_identity

            identity = get_git_build_identity()
        except Exception:
            identity = None
        suffix = f" · {identity}" if identity else ""
        return f"Hermes Agent v{HERMES_VERSION}{suffix}"

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        self._guard_owned_session(session_id, "session/set_model")
        state = self.session_manager.get_session(session_id)
        if state:
            current_provider = getattr(state.agent, "provider", None)
            requested_provider, resolved_model = self._resolve_model_selection(
                model_id,
                current_provider or "openrouter",
            )
            provider_changed = bool(current_provider and requested_provider != current_provider)
            current_base_url = None if provider_changed else getattr(state.agent, "base_url", None)
            current_api_mode = None if provider_changed else getattr(state.agent, "api_mode", None)
            new_agent = self.session_manager._make_agent(
                session_id=session_id,
                cwd=state.cwd,
                model=resolved_model,
                requested_provider=requested_provider,
                base_url=current_base_url,
                api_mode=current_api_mode,
            )
            if not await self._restore_model_switch_mcp_servers(state, new_agent):
                _dispose_replaced_agent(new_agent)
                raise RuntimeError("Failed to restore ACP MCP servers for model switch")
            self._commit_model_switch(state, model=resolved_model, new_agent=new_agent)
            logger.info(
                "Session %s: model switched to %s via provider %s",
                session_id,
                resolved_model,
                requested_provider,
            )
            # The caller's model-switch acknowledgement must describe the
            # replacement agent, not the pre-switch snapshot.  In managed
            # Switchboard mode this is the evidence that the trusted MCP
            # surface survived the mandatory rebuild.
            return SetSessionModelResponse(_meta=self._session_meta(state))
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        self._guard_owned_session(session_id, "session/set_mode")
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        normalized_mode = str(mode_id or "").strip()
        if normalized_mode not in self._MODE_TO_EDIT_APPROVAL_POLICY:
            normalized_mode = self._MODE_DEFAULT
        setattr(state, "mode", normalized_mode)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, normalized_mode)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        self._guard_owned_session(session_id, "session/set_config_option")
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        if str(config_id) == self._EDIT_APPROVAL_POLICY_CONFIG_ID:
            mode = self._EDIT_APPROVAL_POLICY_TO_MODE.get(str(value), self._MODE_DEFAULT)
            setattr(state, "mode", mode)
        else:
            options = getattr(state, "config_options", None)
            if not isinstance(options, dict):
                options = {}
            options[str(config_id)] = value
            setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from acp.schema import AgentThoughtChunk  # noqa: F401,E402
from acp.schema import AudioContentBlock  # noqa: F401,E402
from acp.schema import AvailableCommand  # noqa: F401,E402
from acp.schema import AvailableCommandsUpdate  # noqa: F401,E402
from acp.schema import BlobResourceContents  # noqa: F401,E402
from acp.schema import EmbeddedResourceContentBlock  # noqa: F401,E402
from acp.schema import ImageContentBlock  # noqa: F401,E402
from pathlib import Path  # noqa: F401,E402
from acp.schema import ResourceContentBlock  # noqa: F401,E402
from acp.schema import TextResourceContents  # noqa: F401,E402
from acp.schema import UnstructuredCommandInput  # noqa: F401,E402
import base64  # noqa: F401,E402
import json  # noqa: F401,E402
from urllib.parse import unquote  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ACP_MAX_MODELS_PER_PROVIDER': ('acp_adapter.model_catalog', 'ACP_MAX_MODELS_PER_PROVIDER'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
