"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import base64
import contextvars
import json
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Optional
from urllib.parse import unquote, urlparse

import acp
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerHttp,
    McpServerSse,
    McpServerStdio,
    ModelInfo,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionModelState,
    SessionResumeCapabilities,
    SessionInfo,
    TextContentBlock,
    TextResourceContents,
    UnstructuredCommandInput,
    Usage,
    UsageUpdate,
    UserMessageChunk,
)

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider
from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    flush_open_tool_calls,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.provenance import session_provenance_meta
from acp_adapter.session import (
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
)
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)
from agent.interrupt_compat import request_hard_interrupt
from tools.approval import (
    reset_hermes_interactive_context,
    set_hermes_interactive_context,
)

logger = logging.getLogger(__name__)

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


def _named_custom_provider_catalogs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Return ``(slug, label, [(model_id, description), ...])`` for named endpoints.

    Covers both the v12 ``providers:`` mapping and the legacy
    ``custom_providers:`` list.  These endpoints never appear in canonical
    provider enumeration, so without this the ACP model selector hides every
    named endpoint that the TUI ``/model`` picker already renders (#47039
    implemented named-endpoint rows for the TUI surface only).

    Model lists come from the entry's declared models (``default_model`` +
    ``models``), refreshed from the endpoint's live ``/models`` listing when a
    credential is available and ``discover_models`` is not disabled.  Declared
    models are kept even when live discovery fails — some OpenAI-compatible
    endpoints (e.g. Bedrock Mantle Responses) expose no ``/models`` route at
    all yet serve the declared models fine.

    Slugs use the ``custom:<name>`` shape that ``parse_model_input`` and
    ``resolve_runtime_provider`` already resolve, so encoded choice ids
    (``custom:<name>:<model>``) round-trip through ``set_session_model``
    unchanged.
    """
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers,
            is_provider_enabled,
            load_config,
        )
        from hermes_cli.models import cached_fetch_api_models
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return []

    try:
        cfg = load_config()
        entries = get_compatible_custom_providers(cfg)
    except Exception:
        logger.debug("Could not load named custom providers", exc_info=True)
        return []

    # ``get_compatible_custom_providers`` drops the ``enabled`` flag during
    # normalization, so collect explicitly disabled provider keys from the
    # raw config and skip their entries below.
    disabled_keys: set[str] = set()
    raw_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if isinstance(raw_providers, dict):
        for raw_key, raw_entry in raw_providers.items():
            if isinstance(raw_entry, dict) and not is_provider_enabled(raw_entry):
                disabled_keys.add(str(raw_key).strip().lower())

    catalogs: list[tuple[str, str, list[tuple[str, str]]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider_key = str(entry.get("provider_key", "") or "").strip()
        if provider_key.lower() in disabled_keys:
            continue
        name = str(entry.get("name", "") or "").strip()
        base_url = str(entry.get("base_url", "") or "").strip()
        if not name or not base_url:
            continue
        slug = custom_provider_slug(name, provider_key)

        api_key = str(entry.get("api_key", "") or "").strip()
        if not api_key:
            key_env = str(entry.get("key_env", "") or "").strip()
            api_key = os.environ.get(key_env, "").strip() if key_env else ""

        declared: list[str] = []
        default_model = str(entry.get("model", "") or "").strip()
        if default_model:
            declared.append(default_model)
        models_cfg = entry.get("models")
        if isinstance(models_cfg, dict):
            for mid in models_cfg:
                mid = str(mid or "").strip()
                if mid and mid not in declared:
                    declared.append(mid)

        if not api_key and not declared:
            # No credential to discover with and nothing declared:
            # not addressable from the selector.
            continue

        model_ids = list(declared)
        discover = entry.get("discover_models", True)
        if isinstance(discover, str):
            discover = discover.lower() not in {"false", "no", "0"}
        if discover and api_key:
            try:
                live = cached_fetch_api_models(
                    api_key, base_url, api_mode=entry.get("api_mode")
                )
            except Exception:
                live = None
            if live:
                model_ids = declared + [m for m in live if m not in declared]

        if not model_ids:
            continue
        catalogs.append((slug, name, [(mid, "") for mid in model_ids]))

    return catalogs
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
                    messages,
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


def _dispose_replaced_agent(agent: Any) -> None:
    """Release agent-owned clients without touching shared session resources."""
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

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# Server-side page size for list_sessions. The ACP ListSessionsRequest schema
# does not expose a client-side limit, so this is a fixed cap that clients
# paginate against using `cursor` / `next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50
# Per-provider cap for the ACP model selector. ACP clients (Zed, Buzz) render
# the whole `availableModels` array in one dropdown, so an unbounded
# cross-provider catalog degrades the picker. Mirrors the cap the MoA picker
# already uses (`hermes_cli/moa_cmd.py`). This bounds each provider's row, not
# the total; aggregator providers stay intentionally uncapped inside the shared
# inventory, and the current model is always kept via the fallback insert below.
ACP_MAX_MODELS_PER_PROVIDER = 200
_MAX_ACP_RESOURCE_BYTES = 512 * 1024
_TEXT_RESOURCE_MIME_PREFIXES = ("text/",)
_TEXT_RESOURCE_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/typescript",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
}


def _resource_display_name(uri: str, name: str | None = None, title: str | None = None) -> str:
    """Human-readable attachment name for prompt context."""
    raw_name = (name or "").strip()
    raw_title = (title or "").strip()
    if raw_title and raw_name and raw_title != raw_name:
        return f"{raw_title} ({raw_name})"
    if raw_title:
        return raw_title
    if raw_name:
        return raw_name
    parsed = urlparse(uri)
    candidate = parsed.path if parsed.scheme else uri
    return Path(unquote(candidate)).name or uri or "resource"


def _is_text_resource(mime_type: str | None) -> bool:
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if not mime:
        return False
    return mime.startswith(_TEXT_RESOURCE_MIME_PREFIXES) or mime in _TEXT_RESOURCE_MIME_TYPES


def _is_image_resource(mime_type: str | None) -> bool:
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    return mime.startswith("image/")


def _guess_image_mime_from_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }.get(suffix)


def _image_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _path_from_file_uri(uri: str) -> Path | None:
    """Convert local file URIs/paths from ACP clients into a readable Path.

    Zed may send POSIX file URIs from Linux/WSL workspaces or Windows-ish paths
    when launched through wsl.exe. Translate the common Windows drive form to
    /mnt/<drive>/... so Hermes running in WSL can read it.
    """
    raw = (uri or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        return None

    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            return None
        path_text = unquote(parsed.path or "")
    else:
        path_text = unquote(raw)

    # file:///C:/Users/... or C:\Users\...
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":" and path_text[1].isalpha():
        drive = path_text[1].lower()
        rest = path_text[3:].lstrip("/\\").replace("\\", "/")
        return Path("/mnt") / drive / rest
    if len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha():
        drive = path_text[0].lower()
        rest = path_text[2:].lstrip("/\\").replace("\\", "/")
        return Path("/mnt") / drive / rest

    return Path(path_text)


def _decode_text_bytes(data: bytes, mime_type: str | None) -> str | None:
    """Decode resource bytes if they are probably text; return None for binary."""
    if b"\x00" in data and not _is_text_resource(mime_type):
        return None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _format_resource_text(
    *,
    uri: str,
    body: str,
    name: str | None = None,
    title: str | None = None,
    note: str | None = None,
) -> str:
    display = _resource_display_name(uri, name=name, title=title)
    header = f"[Attached file: {display}]"
    if note:
        header += f" ({note})"
    return f"{header}\nURI: {uri}\n\n{body}"


def _resource_link_to_parts(block: ResourceContentBlock) -> list[dict[str, Any]]:
    """Convert an ACP resource_link block to OpenAI content parts.

    Returns a list of {"type": "text", ...} and/or {"type": "image_url", ...}
    parts. Image resources produce an image_url part with a small text header
    so the model knows which attachment it is. Non-image resources return a
    single text part with the inlined file body (or a binary-omit note).
    """
    uri = str(getattr(block, "uri", "") or "").strip()
    if not uri:
        return []

    name = str(getattr(block, "name", "") or "").strip() or None
    title = str(getattr(block, "title", "") or "").strip() or None
    mime_type = str(getattr(block, "mime_type", "") or "").strip() or None
    path = _path_from_file_uri(uri)

    if path is None:
        return [{
            "type": "text",
            "text": _format_resource_text(
                uri=uri,
                name=name,
                title=title,
                body="[Resource link only; Hermes cannot read non-file ACP resource URIs directly.]",
            ),
        }]

    # Image files: emit a short text header + image_url data URL so vision
    # models can see the attachment instead of a "binary omitted" note.
    image_mime = mime_type if _is_image_resource(mime_type) else _guess_image_mime_from_path(path)
    if image_mime and _is_image_resource(image_mime):
        try:
            size = path.stat().st_size
            if size > _MAX_ACP_RESOURCE_BYTES:
                return [{
                    "type": "text",
                    "text": _format_resource_text(
                        uri=uri,
                        name=name,
                        title=title,
                        body=f"[Image too large to inline: {size} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                    ),
                }]
            with path.open("rb") as fh:
                data = fh.read()
        except OSError as exc:
            logger.warning("ACP image resource read failed: %s", uri, exc_info=True)
            return [{
                "type": "text",
                "text": _format_resource_text(
                    uri=uri,
                    name=name,
                    title=title,
                    body=f"[Could not read attached image: {exc}]",
                ),
            }]
        display = _resource_display_name(uri, name=name, title=title)
        return [
            {"type": "text", "text": f"[Attached image: {display}]\nURI: {uri}"},
            {"type": "image_url", "image_url": {"url": _image_data_url(data, image_mime)}},
        ]

    try:
        size = path.stat().st_size
        read_size = min(size, _MAX_ACP_RESOURCE_BYTES)
        with path.open("rb") as fh:
            data = fh.read(read_size)
        text = _decode_text_bytes(data, mime_type)
        if text is None:
            return [{
                "type": "text",
                "text": _format_resource_text(
                    uri=uri,
                    name=name,
                    title=title,
                    body=f"[Binary file omitted: {size} bytes, mime={mime_type or 'unknown'}]",
                ),
            }]
        note = None
        if size > _MAX_ACP_RESOURCE_BYTES:
            note = f"truncated to {_MAX_ACP_RESOURCE_BYTES} of {size} bytes"
        return [{
            "type": "text",
            "text": _format_resource_text(uri=uri, name=name, title=title, body=text, note=note),
        }]
    except OSError as exc:
        logger.warning("ACP resource read failed: %s", uri, exc_info=True)
        return [{
            "type": "text",
            "text": _format_resource_text(
                uri=uri,
                name=name,
                title=title,
                body=f"[Could not read attached file: {exc}]",
            ),
        }]


def _embedded_resource_to_parts(block: EmbeddedResourceContentBlock) -> list[dict[str, Any]]:
    resource = getattr(block, "resource", None)
    if resource is None:
        return []

    uri = str(getattr(resource, "uri", "") or "").strip()
    mime_type = str(getattr(resource, "mime_type", "") or "").strip() or None

    if isinstance(resource, TextResourceContents):
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=resource.text)}]

    if isinstance(resource, BlobResourceContents):
        blob = resource.blob or ""
        try:
            data = base64.b64decode(blob, validate=True)
        except Exception:
            data = blob.encode("utf-8", errors="replace")

        # Image blobs go through as image_url so vision models can see them.
        if _is_image_resource(mime_type):
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                return [{
                    "type": "text",
                    "text": _format_resource_text(
                        uri=uri,
                        body=f"[Embedded image too large to inline: {len(data)} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                    ),
                }]
            display = _resource_display_name(uri)
            return [
                {"type": "text", "text": f"[Attached image: {display}]" + (f"\nURI: {uri}" if uri else "")},
                {"type": "image_url", "image_url": {"url": _image_data_url(data, mime_type or "image/png")}},
            ]

        text = _decode_text_bytes(data[:_MAX_ACP_RESOURCE_BYTES], mime_type)
        if text is None:
            body = f"[Binary embedded file omitted: {len(data)} bytes, mime={mime_type or 'unknown'}]"
        else:
            body = text
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                body += f"\n\n[Truncated to {_MAX_ACP_RESOURCE_BYTES} of {len(data)} bytes]"
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=body)}]

    text = getattr(resource, "text", None)
    if text:
        return [{"type": "text", "text": _format_resource_text(uri=uri, body=str(text))}]
    return []


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks for display/commands."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts)


def _image_block_to_openai_part(block: ImageContentBlock) -> dict[str, Any] | None:
    """Convert an ACP image content block to OpenAI-style multimodal content."""
    data = str(getattr(block, "data", "") or "").strip()
    uri = str(getattr(block, "uri", "") or "").strip()
    mime_type = str(getattr(block, "mime_type", "") or "image/png").strip() or "image/png"

    if data:
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
    elif uri:
        url = uri
    else:
        return None

    return {"type": "image_url", "image_url": {"url": url}}


def _content_blocks_to_openai_user_content(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str | list[dict[str, Any]]:
    """Convert ACP prompt blocks into a Hermes/OpenAI-compatible user content payload."""
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            continue
        if isinstance(block, ImageContentBlock):
            image_part = _image_block_to_openai_part(block)
            if image_part is not None:
                parts.append(image_part)
            continue
        if isinstance(block, ResourceContentBlock):
            resource_parts = _resource_link_to_parts(block)
            for part in resource_parts:
                parts.append(part)
                if part.get("type") == "text":
                    text_parts.append(part["text"])
            continue
        if isinstance(block, EmbeddedResourceContentBlock):
            resource_parts = _embedded_resource_to_parts(block)
            for part in resource_parts:
                parts.append(part)
                if part.get("type") == "text":
                    text_parts.append(part["text"])
            continue

    if not parts:
        return _extract_text(prompt)

    # Keep pure text prompts as strings so slash-command handling and text-only
    # providers keep the exact legacy path. Switch to structured content only
    # when an actual non-text block is present.
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(text_parts)

    return parts


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _SLASH_COMMANDS = {
        "help": "Show available commands",
        "model": "Show or change current model",
        "tools": "List available tools",
        "context": "Show conversation context info",
        "reset": "Clear conversation history",
        "compress": "Compress conversation context",
        "steer": "Inject guidance into the currently running agent turn",
        "queue": "Queue a prompt to run after the current turn finishes",
        "version": "Show Hermes version",
    }

    _ADVERTISED_COMMANDS = (
        {
            "name": "help",
            "description": "List available commands",
        },
        {
            "name": "model",
            "description": "Show current model and provider, or switch models",
            "input_hint": "model name to switch to",
        },
        {
            "name": "tools",
            "description": "List available tools with descriptions",
        },
        {
            "name": "context",
            "description": "Show conversation message counts by role",
        },
        {
            "name": "reset",
            "description": "Clear conversation history",
        },
        {
            "name": "compress",
            "description": "Compress conversation context",
        },
        {
            "name": "steer",
            "description": "Inject guidance into the currently running agent turn",
            "input_hint": "guidance for the active turn",
        },
        {
            "name": "queue",
            "description": "Queue a prompt to run after the current turn finishes",
            "input_hint": "prompt to run next",
        },
        {
            "name": "version",
            "description": "Show Hermes version",
        },
    )

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"
    _MODE_ACCEPT_EDITS = "accept_edits"
    _MODE_DONT_ASK = "dont_ask"
    _MODE_TO_EDIT_APPROVAL_POLICY = {
        _MODE_DEFAULT: "ask",
        _MODE_ACCEPT_EDITS: "workspace_session",
        _MODE_DONT_ASK: "session",
    }
    _EDIT_APPROVAL_POLICY_TO_MODE = {
        value: key for key, value in _MODE_TO_EDIT_APPROVAL_POLICY.items()
    }

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None
        self._delegation_watcher_task: Optional[asyncio.Task] = None

    # ---- Background delegation completions -----------------------------------

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


    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Return ACP session modes while preserving Zed's separate model picker.

        Zed renders ``config_options`` in the prominent selector slot where the
        model picker was visible. Claude/Codex expose policy-like controls as ACP
        modes, which coexist with the model picker, so Hermes maps edit approval
        policy onto modes instead of advertising config options.
        """

        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODE_TO_EDIT_APPROVAL_POLICY:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[
                SessionMode(
                    id=self._MODE_DEFAULT,
                    name="Default",
                    description="Ask before edits.",
                ),
                SessionMode(
                    id=self._MODE_ACCEPT_EDITS,
                    name="Accept Edits",
                    description="Auto-allow workspace and /tmp edits; still asks for sensitive paths.",
                ),
                SessionMode(
                    id=self._MODE_DONT_ASK,
                    name="Don't Ask",
                    description="Auto-allow file edits for this session except sensitive paths.",
                ),
            ],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

    @staticmethod
    def _encode_model_choice(provider: str | None, model: str | None) -> str:
        """Encode a model selection so ACP clients can keep provider context."""
        raw_model = str(model or "").strip()
        if not raw_model:
            return ""
        raw_provider = str(provider or "").strip().lower()
        if not raw_provider:
            return raw_model
        return f"{raw_provider}:{raw_model}"

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Return authenticated providers and their models for ACP clients.

        The shared Hermes inventory is also used by ``hermes model``, the TUI,
        and the dashboard. Keeping ACP on that substrate prevents its selector
        from silently collapsing to the current provider's curated list.
        """
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"

        try:
            from hermes_cli.inventory import build_models_payload, load_picker_context
            from hermes_cli.models import normalize_provider, provider_label

            normalized_provider = normalize_provider(provider)
            context = load_picker_context().with_overrides(
                current_provider=normalized_provider,
                current_model=model,
                current_base_url=str(getattr(state.agent, "base_url", "") or ""),
            )
            payload = build_models_payload(
                context,
                explicit_only=True,
                include_unconfigured=False,
                picker_hints=False,
                canonical_order=True,
                pricing=False,
                capabilities=False,
                refresh=False,
                probe_custom_providers=False,
                probe_current_custom_provider=False,
                max_models=ACP_MAX_MODELS_PER_PROVIDER,
            )

            available_models: list[ModelInfo] = []
            seen_ids: set[str] = set()
            for row in payload.get("providers") or []:
                row_provider = normalize_provider(str(row.get("slug") or "").strip())
                if not row_provider:
                    continue
                provider_name = str(row.get("name") or "").strip() or provider_label(
                    row_provider
                )
                for model_entry in row.get("models") or []:
                    if isinstance(model_entry, dict):
                        rendered_model = str(
                            model_entry.get("id")
                            or model_entry.get("model")
                            or model_entry.get("name")
                            or ""
                        ).strip()
                    else:
                        rendered_model = str(model_entry or "").strip()
                    if not rendered_model:
                        continue
                    choice_id = self._encode_model_choice(row_provider, rendered_model)
                    if choice_id in seen_ids:
                        continue
                    is_current = (
                        row_provider == normalized_provider and rendered_model == model
                    )
                    description = f"Provider: {provider_name}"
                    if is_current:
                        description += " • current"
                    available_models.append(
                        ModelInfo(
                            model_id=choice_id,
                            name=f"{provider_name} · {rendered_model}",
                            description=description,
                        )
                    )
                    seen_ids.add(choice_id)

            # Named user-defined endpoints (providers: / custom_providers:)
            # are invisible to canonical provider enumeration — append them
            # so editor clients can select them like the TUI /model picker.
            for named_slug, named_label, named_catalog in _named_custom_provider_catalogs():
                for named_model, named_desc in named_catalog:
                    named_choice = self._encode_model_choice(named_slug, named_model)
                    if not named_choice or named_choice in seen_ids:
                        continue
                    named_parts = [f"Provider: {named_label}"]
                    if named_desc:
                        named_parts.append(str(named_desc).strip())
                    if named_slug == normalized_provider and named_model == model:
                        named_parts.append("current")
                    available_models.append(
                        ModelInfo(
                            model_id=named_choice,
                            name=named_model,
                            description=" • ".join(part for part in named_parts if part),
                        )
                    )
                    seen_ids.add(named_choice)

            current_model_id = self._encode_model_choice(normalized_provider, model)
            if current_model_id and current_model_id not in seen_ids:
                provider_name = provider_label(normalized_provider)
                available_models.insert(
                    0,
                    ModelInfo(
                        model_id=current_model_id,
                        name=f"{provider_name} · {model}",
                        description=f"Provider: {provider_name} • current",
                    ),
                )

            if available_models:
                return SessionModelState(
                    available_models=available_models,
                    current_model_id=current_model_id or available_models[0].model_id,
                )
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None

        fallback_choice = self._encode_model_choice(provider, model)
        return SessionModelState(
            available_models=[ModelInfo(model_id=fallback_choice, name=model)],
            current_model_id=fallback_choice,
        )

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id."""
        target_provider = current_provider
        new_model = raw_model.strip()

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

    def _commit_model_switch(
        self,
        state: SessionState,
        *,
        model: str,
        new_agent: Any,
    ) -> None:
        """Persist a replacement before retiring the previous live agent."""
        previous_model = state.model
        previous_agent = state.agent
        state.model = model
        state.agent = new_agent
        if not self.session_manager.save_session(state.session_id):
            state.model = previous_model
            state.agent = previous_agent
            self.session_manager.save_session(state.session_id)
            _dispose_replaced_agent(new_agent)
            raise RuntimeError("Failed to persist ACP model switch")
        _dispose_replaced_agent(previous_agent)

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """Build ACP native context-usage data for clients like Zed.

        Zed's circular context indicator is driven by ACP ``usage_update``
        session updates: ``size`` is the model context window and ``used`` is
        the current request pressure.  Hermes estimates ``used`` from the same
        buckets it sends to providers: system prompt, conversation history, and
        tool schemas.
        """
        agent = state.agent
        compressor = getattr(agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None

        try:
            from agent.model_metadata import estimate_request_tokens_rough

            used = estimate_request_tokens_rough(
                state.history,
                system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
                tools=getattr(agent, "tools", None) or None,
            )
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)

        return UsageUpdate(
            session_update="usage_update",
            size=max(size, 0),
            used=max(used, 0),
        )

    async def _send_usage_update(self, state: SessionState) -> None:
        """Send ACP native context usage to the connected client."""
        if not self._conn:
            return
        update = self._build_usage_update(state)
        if update is None:
            return
        try:
            await self._conn.session_update(
                session_id=state.session_id,
                update=update,
            )
        except Exception:
            logger.warning(
                "Failed to send ACP usage update for session %s",
                state.session_id,
                exc_info=True,
            )

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
        self,
        acp_session_id: str,
        current_hermes_session_id: str,
        previous_hermes_session_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(),
                acp_session_id,
                current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug(
                "Could not build ACP session provenance for %s", acp_session_id, exc_info=True
            )
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
        """Send ACP native session metadata after Hermes changes it.

        When the internal Hermes head rotated (e.g. compression-driven session
        split during a turn), pass ``previous_hermes_session_id`` so the
        attached ``_meta.hermes.sessionProvenance`` flags the rotation reason.
        """
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
        # The `sessions` table does not have an `updated_at` column (see
        # hermes_state.py schema — only started_at/ended_at). Use "now" as
        # the updated_at since we're emitting this notification precisely
        # because the title was just refreshed.
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
            await self._conn.session_update(
                session_id=session_id,
                update=update,
            )
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
        """Refresh the agent's tool snapshot when background MCP discovery lands late.

        ACP entry.py starts MCP tool discovery in a background daemon thread so a
        slow/dead configured server can't block ``asyncio.run()``.  ``_make_agent``
        briefly joins that thread (``wait_for_mcp_discovery``, bounded ~1.5s) so
        already-spawning fast servers land in the snapshot — but a server slower
        than the bound lands *after* the agent is built, leaving its tools absent
        for the whole session.

        This schedules an off-critical-path daemon that waits for discovery to
        finish (bounded 30s), then rebuilds the snapshot via the shared
        ``refresh_agent_mcp_tools`` helper — the same rebuild ``/reload-mcp``
        performs, but automatic.  Mirrors the TUI late-refresh (PR #48403).

        Cache safety: the rebuild only runs while the session is still
        pre-first-turn (no API call made yet → nothing cached to invalidate).
        Once the user has sent a message we leave the snapshot frozen rather
        than break the cached prompt prefix mid-conversation; servers that land
        later are picked up cache-safely by the between-turns prologue refresh
        (``agent/turn_context.py``) at the next turn boundary.  The marginal
        value of this pre-first-turn daemon is therefore freshness in the
        window [session created → first message] — e.g. the "Available tools"
        listing a client may request before the first prompt.
        No-op when discovery already finished, when the join times out, when the
        registry was unchanged, or when the session was closed while waiting.
        """
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight
        except Exception:
            return
        if not mcp_discovery_in_flight():
            return

        import threading

        agent = state.agent
        session_id = state.session_id

        def _wait_then_refresh() -> None:
            try:
                from hermes_cli.mcp_startup import join_mcp_discovery

                if not join_mcp_discovery(timeout=30.0):
                    return

                # Session may have been closed while we waited.  In-memory-only
                # lookup on purpose: ``get_session()`` falls through to a DB
                # restore that builds a whole new AIAgent as a side effect just
                # to decide "no-op" here (the TUI equivalent also checks its
                # in-memory dict only).
                with self.session_manager._lock:
                    current = self.session_manager._sessions.get(session_id)
                if current is None or current.agent is not agent:
                    return

                # Cache safety: never rebuild the tool list once the conversation
                # has started — that would invalidate the cached prompt prefix.
                # Serialized with turn start: ``prompt()`` flips ``is_running``
                # under ``runtime_lock`` before dispatching, so holding it here
                # (and bailing when a turn is already running) closes the window
                # where the guard passes but the first prompt starts before the
                # refresh publishes — which would swap ``tools=`` mid-turn and
                # break the just-created cache prefix.
                with current.runtime_lock:
                    if current.is_running:
                        return
                    if (
                        int(getattr(agent, "_user_turn_count", 0) or 0) > 0
                        or int(getattr(agent, "_api_call_count", 0) or 0) > 0
                    ):
                        return

                    from tools.mcp_tool import refresh_agent_mcp_tools

                    added = refresh_agent_mcp_tools(agent, quiet_mode=True)
                if added:
                    logger.info(
                        "Session %s: late MCP refresh added %d tools: %s",
                        session_id,
                        len(added),
                        ", ".join(sorted(added)),
                    )
            except Exception:
                logger.debug(
                    "Session %s: late MCP refresh failed",
                    session_id,
                    exc_info=True,
                )

        threading.Thread(
            target=_wait_then_refresh,
            name=f"acp-mcp-late-refresh-{session_id}",
            daemon=True,
        ).start()

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        resolved_protocol_version = (
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION
        )
        auth_methods = build_auth_methods()

        client_name = client_info.name if client_info else "unknown"
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_name,
            resolved_protocol_version,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only accept authenticate() calls whose method_id matches the
        # provider we advertised in initialize(). Without this check,
        # authenticate() would acknowledge any method_id as long as the
        # server has provider credentials configured — harmless under
        # Hermes' threat model (ACP is stdio-only, local-trust), but poor
        # API hygiene and confusing if ACP ever grows multi-method auth.
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth launches Hermes setup/model selection out-of-band.
            # Only report success once that flow has produced usable runtime
            # credentials for the normal ACP session.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    @staticmethod
    def _flatten_history_text(value: Any) -> str:
        """Normalize a persisted text-or-text-parts value into a single string.

        OpenAI-style assistant content (and provider reasoning fields) can arrive
        as either a scalar string or a list of ``{"text": ...}`` /
        ``{"type": "text", "content": ...}`` parts. Whitespace-only inputs
        collapse to an empty string so callers can treat ``""`` as "nothing to
        emit".
        """
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

    @classmethod
    def _history_message_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable text from a persisted OpenAI-style message."""
        return cls._flatten_history_text(message.get("content"))

    @classmethod
    def _history_reasoning_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable reasoning/thought text from a persisted assistant message.

        Returns the first non-empty value among ``reasoning_content`` (the
        canonical field used by DeepSeek / Moonshot and the post-#16892
        chat-completions normalizer) and ``reasoning`` (used by the codex
        event projector and several other transports). Both keys are
        actively written by live code paths, so neither branch is
        deprecated — they cover different transports rather than old vs.
        new sessions.
        """
        for key in ("reasoning_content", "reasoning"):
            text = cls._flatten_history_text(message.get(key))
            if text:
                return text
        return ""

    @staticmethod
    def _history_summary_meta(message: dict[str, Any], text: str) -> dict[str, Any] | None:
        """Build the ``_meta`` payload for a replayed compaction summary.

        Compaction summaries are persisted as ordinary history messages —
        standalone handoffs under ``role="user"`` OR ``role="assistant"``
        (the compressor picks whichever role keeps alternation valid), and
        merge-into-tail messages where the summary is appended after the
        first preserved tail message's real content. Without a wire flag,
        ACP frontends render all of these as ordinary turns.

        Two distinct keys under ``_meta.hermes`` (ACP's extensibility
        channel), so clients cannot accidentally hide real content:

        * ``compactionSummary: true`` — the entire chunk is the handoff
          summary. Safe to restyle or collapse wholesale.
        * ``containsCompactionSummary: true`` — a merged-tail message: real
          preserved turn content followed by the summary. Clients may style
          it, but collapsing the whole chunk would hide the preserved
          content, hence the separate key.

        Detection honors the in-process ``_compressed_summary`` flag and
        falls back to content classification, so it also works for a
        DB-reloaded session that lost the in-memory flag.
        """
        kind = ContextCompressor.classify_summary_content(text)
        if kind is None and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            # Flagged in-process but content didn't classify (e.g. future
            # prefix drift): treat as a standalone summary — the flag is only
            # ever set on summary-bearing messages.
            kind = "standalone"
        if kind == "standalone":
            return {"hermes": {"compactionSummary": True}}
        if kind == "merged":
            return {"hermes": {"containsCompactionSummary": True}}
        return None

    @staticmethod
    def _history_message_update(
        *,
        role: str,
        text: str,
        field_meta: dict[str, Any] | None = None,
    ) -> UserMessageChunk | AgentMessageChunk | None:
        """Build an ACP history replay update for a user/assistant message."""
        block = TextContentBlock(type="text", text=text)
        if role == "user":
            return UserMessageChunk(
                session_update="user_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        if role == "assistant":
            return AgentMessageChunk(
                session_update="agent_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        return None

    @staticmethod
    def _history_thought_update(text: str) -> AgentThoughtChunk:
        """Build an ACP history replay update for an assistant thought."""
        return acp.update_agent_thought_text(text)

    @staticmethod
    def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract function name/arguments from an OpenAI-style tool_call."""
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
        raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception:
                parsed = {"raw": raw_args}
            raw_args = parsed
        if not isinstance(raw_args, dict):
            raw_args = {}
        return name, raw_args

    @staticmethod
    def _history_tool_call_id(tool_call: dict[str, Any]) -> str:
        """Return the stable provider tool call id for ACP history replay."""
        return str(
            tool_call.get("id")
            or tool_call.get("call_id")
            or tool_call.get("tool_call_id")
            or ""
        ).strip()

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay persisted user/assistant history during session/load or session/resume.

        Invoked inline (``await``) from both ``load_session`` and
        ``resume_session`` so that spec-compliant ACP clients receive the
        full transcript within the request's lifetime — see the comment at
        the call sites for the rationale and prior-art citations.

        Replays the conversation as user/assistant chunks, thinking-mode
        thought chunks, plus reconstructed tool-call start/completion
        notifications. Merely restoring server-side state makes Hermes
        remember context, but leaves the editor looking like a clean thread.
        """
        if not self._conn or not state.history:
            return

        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        async def _send(update: Any) -> bool:
            try:
                await self._conn.session_update(session_id=state.session_id, update=update)
                return True
            except Exception:
                logger.warning(
                    "Failed to replay ACP history for session %s",
                    state.session_id,
                    exc_info=True,
                )
                return False

        for message in state.history:
            role = str(message.get("role") or "")

            if role == "user":
                text = self._history_message_text(message)
                if text:
                    update = self._history_message_update(
                        role=role,
                        text=text,
                        field_meta=self._history_summary_meta(message, text),
                    )
                    if update is not None and not await _send(update):
                        return
                continue

            if role == "assistant":
                thought = self._history_reasoning_text(message)
                if thought and not await _send(self._history_thought_update(thought)):
                    return

                text = self._history_message_text(message)
                if text:
                    update = self._history_message_update(
                        role=role,
                        text=text,
                        field_meta=self._history_summary_meta(message, text),
                    )
                    if update is not None and not await _send(update):
                        return

                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = self._history_tool_call_id(tool_call)
                        if not tool_call_id:
                            continue
                        tool_name, args = self._history_tool_call_name_args(tool_call)
                        active_tool_calls[tool_call_id] = (tool_name, args)
                        if not await _send(build_tool_start(tool_call_id, tool_name, args)):
                            return
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                tool_name = str(message.get("tool_name") or "").strip()
                function_args: dict[str, Any] | None = None
                if tool_call_id in active_tool_calls:
                    tool_name, function_args = active_tool_calls.pop(tool_call_id)
                if not tool_call_id or not tool_name:
                    continue
                result = message.get("content")
                result_text = result if isinstance(result, str) else None
                if not await _send(
                    build_tool_complete(
                        tool_call_id,
                        tool_name,
                        result=result_text,
                        function_args=function_args,
                    )
                ):
                    return
                if tool_name == "todo":
                    plan_update = _build_plan_update_from_todo_result(result_text)
                    if plan_update is not None and not await _send(plan_update):
                        return

    # ---- Cross-session ownership guard --------------------------------------

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
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List ACP sessions with optional ``cwd`` filtering and cursor pagination.

        ``cwd`` is passed through to ``SessionManager.list_sessions`` which already
        normalizes and filters by working directory. ``cursor`` is a ``session_id``
        previously returned as ``next_cursor``; results resume after that entry.
        Server-side page size is capped at ``_LIST_SESSIONS_PAGE_SIZE``; when more
        results remain, ``next_cursor`` is set to the last returned ``session_id``.
        """
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                # Unknown cursor -> empty page (do not fall back to full list).
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        infos = infos[:_LIST_SESSIONS_PAGE_SIZE]

        sessions = []
        for s in infos:
            updated_at = s.get("updated_at")
            if updated_at is not None and not isinstance(updated_at, str):
                updated_at = str(updated_at)
            sessions.append(
                SessionInfo(
                    session_id=s["session_id"],
                    cwd=s["cwd"],
                    title=s.get("title"),
                    updated_at=updated_at,
                )
            )

        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

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
            if response_text is not None:
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

        streamed_message = False
        turn_history_baseline_count = len(state.history)

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
            result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
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
        finally:
            # Mark this turn idle before draining queued work so recursive prompt()
            # calls can acquire the session. Queued turns are intentionally run as
            # normal follow-up user prompts, preserving role alternation and history.
            # This reset + the drain below MUST run no matter how the block above
            # exits (success, caught error, or an uncaught exception) so a turn
            # that cooperatively returned from the executor never leaves the
            # session wedged with is_running=True and an undrained queue.
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""

            if keepalive_task is not None:
                keepalive_task.cancel()

            await self._drain_queued_prompts(state, session_id, conn)

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

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands: list[AvailableCommand] = []
        for spec in cls._ADVERTISED_COMMANDS:
            input_hint = spec.get("input_hint")
            commands.append(
                AvailableCommand(
                    name=spec["name"],
                    description=spec["description"],
                    input=UnstructuredCommandInput(hint=input_hint)
                    if input_hint
                    else None,
                )
            )
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._available_commands(),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to advertise ACP slash commands for session %s",
                session_id,
                exc_info=True,
            )

    def _schedule_available_commands_update(self, session_id: str) -> None:
        """Send the command advertisement after the session response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(
            asyncio.create_task, self._send_available_commands_update(session_id)
        )

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command and return the response text.

        Returns ``None`` for unrecognized commands so they fall through
        to the LLM (the user may have typed ``/something`` as prose).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "tools": self._cmd_tools,
            "context": self._cmd_context,
            "reset": self._cmd_reset,
            "compress": self._cmd_compress,
            "steer": self._cmd_steer,
            "queue": self._cmd_queue,
            "version": self._cmd_version,
        }.get(cmd)

        if handler is None:
            return None  # not a known command — let the LLM handle it

        # Slash handlers run on the event-loop thread, OUTSIDE the per-turn
        # contextvars.copy_context() that pins the session cwd for the agent
        # call. ``/compress`` and ``/model`` reach code that REBUILDS the
        # system prompt (agent._build_system_prompt -> resolve_agent_cwd), so
        # an unpinned handler bakes the Hermes install tree into the session's
        # cached prompt — persisted, and therefore poisoning every later turn
        # even though the turn itself is pinned. Pin inside a fresh context so
        # the write can't leak into other concurrent ACP sessions and needs no
        # teardown.
        def _dispatch() -> str | None:
            try:
                from agent.runtime_cwd import set_session_cwd

                set_session_cwd(state.cwd)
            except Exception:
                logger.debug("Could not pin ACP session cwd for slash command", exc_info=True)
            return handler(args, state)

        try:
            return contextvars.copy_context().run(_dispatch)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        for cmd, desc in self._SLASH_COMMANDS.items():
            lines.append(f"  /{cmd:10s}  {desc}")
        lines.append("")
        lines.append("Unrecognized /commands are sent to the model as normal messages.")
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider = getattr(state.agent, "provider", None) or "openrouter"
        target_provider, new_model = self._resolve_model_selection(args, current_provider)

        new_agent = self.session_manager._make_agent(
            session_id=state.session_id,
            cwd=state.cwd,
            model=new_model,
            requested_provider=target_provider,
        )
        self._commit_model_switch(state, model=new_model, new_agent=new_agent)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            from types import SimpleNamespace
            from agent.memory_manager import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            )
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={
                    tool.get("function", {}).get("name")
                    for tool in tools or []
                    if isinstance(tool, dict)
                },
                enabled_toolsets=toolsets,
                _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = (t.get("function") or {}).get("name", "?")
                desc = (t.get("function") or {}).get("description", "")
                # Truncate long descriptions
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)

        # Count by role.
        roles: dict[str, int] = {}
        for msg in state.history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            from agent.model_metadata import estimate_request_tokens_rough

            system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            tools = getattr(agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                state.history,
                system_prompt=system_prompt,
                tools=tools,
            )
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages"
            if n_messages
            else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0:
            if context_length > 0:
                usage_pct = (approx_tokens / context_length) * 100
                lines.append(
                    f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)"
                )
            else:
                lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0:
            if approx_tokens > 0:
                threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
                remaining = max(threshold_tokens - approx_tokens, 0)
                if approx_tokens >= threshold_tokens:
                    lines.append(
                        f"Compression: due now (threshold ~{threshold_tokens:,}"
                        + (f", {threshold_pct:.0f}%" if threshold_pct else "")
                        + "). Run /compress."
                    )
                else:
                    lines.append(
                        f"Compression: ~{remaining:,} tokens until threshold "
                        f"(~{threshold_tokens:,}"
                        + (f", {threshold_pct:.0f}%" if threshold_pct else "")
                        + ")."
                    )
            else:
                lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        if getattr(agent, "compression_enabled", True) is False:
            lines.append(
                "Auto-compaction is disabled (compression.enabled: false); "
                "/compress still compresses manually."
            )
        else:
            lines.append("Tip: run /compress to compress manually before the threshold.")

        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        reset_failed = False
        try:
            reset_session_state = getattr(state.agent, "reset_session_state", None)
            if callable(reset_session_state):
                reset_session_state()
        except Exception:
            reset_failed = True
            logger.warning("ACP session state reset failed for %s", state.session_id, exc_info=True)
        finally:
            self.session_manager.save_session(state.session_id)
        if reset_failed:
            return "Conversation history cleared. Agent session state reset failed; see logs."
        return "Conversation history cleared."

    def _cmd_compress(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            # No compression_enabled gate: the flag disables *automatic*
            # compaction only; manual /compress must keep working (matches
            # the CLI /compress and gateway handlers).
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            from agent.model_metadata import estimate_request_tokens_rough

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real
            # request pressure, not a transcript-only underestimate (#6217).
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                state.history, system_prompt=_sys_prompt, tools=_tools
            )
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # ACP sessions must keep a stable session id, so avoid the
                # SQLite session-splitting side effect inside _compress_context.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history,
                    getattr(agent, "_cached_system_prompt", "") or "",
                    approx_tokens=approx_tokens,
                    task_id=state.session_id,
                    force=True,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_count = len(state.history)
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            new_tokens = estimate_request_tokens_rough(
                state.history,
                system_prompt=_sys_prompt_after,
                tools=_tools_after,
            )
            return (
                f"Context compressed: {original_count} -> {new_count} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        with state.runtime_lock:
            state.queued_prompts.append(steer_text)
            depth = len(state.queued_prompts)
        return f"No active turn — queued for the next turn. ({depth} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        with state.runtime_lock:
            state.queued_prompts.append(queued_text)
            depth = len(state.queued_prompts)
        return f"Queued for the next turn. ({depth} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- Model switching (ACP protocol method) -------------------------------

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
