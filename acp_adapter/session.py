"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``.  When the editor
reconnects after idle/restart, the ``load_session`` / ``resume_session`` calls
find the persisted session in the database and restore the full conversation
history.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home

import asyncio
import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRANSCRIPT_CORRECTION_POISON_KEY = "acp_transcript_correction_poisoned"


class UnsafeSessionTranscriptError(RuntimeError):
    """Refuse a session whose rejected assistant candidate remains durable."""

    def __init__(self, session_id: str):
        super().__init__(
            "ACP session "
            f"{session_id} is blocked because its transcript could not be "
            "safely corrected; do not resume it until the durable history is repaired"
        )
        self.session_id = session_id


def _translate_acp_cwd(cwd: str) -> str:
    """Translate Windows ACP cwd values when Hermes itself is running in WSL.

    Windows ACP clients can launch ``hermes acp`` inside WSL while still sending
    editor workspaces as Windows drive paths (``E:\\Projects``) or
    ``\\\\wsl.localhost\\`` UNC paths. Store and execute against the POSIX form so
    agents, tools, and persisted ACP sessions all agree on the usable workspace.
    Native Linux/macOS keeps the original cwd unchanged.
    """
    from hermes_constants import translate_cwd_for_wsl_backend

    return translate_cwd_for_wsl_backend(str(cwd))


def _normalize_cwd_for_compare(cwd: str | None) -> str:
    raw = str(cwd or ".").strip()
    if not raw:
        raw = "."
    expanded = os.path.expanduser(raw)

    # Normalize Windows drive paths into the equivalent WSL mount form so
    # ACP history filters match the same workspace across Windows and WSL.
    from hermes_constants import windows_path_to_wsl

    translated = windows_path_to_wsl(expanded)
    if translated is not None:
        expanded = translated
    elif re.match(r"^/mnt/[A-Za-z]/", expanded):
        expanded = f"/mnt/{expanded[5].lower()}/{expanded[7:]}"

    # Resolve symlink aliases so equivalent spellings of the same directory
    # compare equal — macOS reports editor workspaces as ``/var/...`` while
    # sessions get stored under ``/private/var/...`` (and ``/tmp`` vs
    # ``/private/tmp``), which made ACP history filters silently drop a
    # workspace's own sessions. ``os.path.realpath`` is lexical for missing
    # paths (strict=False), so cwds that don't exist on this host — e.g.
    # WSL-translated Windows drives — keep the previous normpath behavior.
    # Ported from PrimeIntellect-ai/prime-agent#628.
    try:
        return os.path.realpath(expanded)
    except OSError:
        return os.path.normpath(expanded)


def _build_session_title(title: Any, preview: Any, cwd: str | None) -> str:
    explicit = str(title or "").strip()
    if explicit:
        return explicit
    preview_text = str(preview or "").strip()
    if preview_text:
        return preview_text
    leaf = os.path.basename(str(cwd or "").rstrip("/\\"))
    return leaf or "New thread"


def _format_updated_at(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _updated_at_sort_key(value: Any) -> float:
    if value is None:
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return float("-inf")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(raw)
        except Exception:
            return float("-inf")


def _acp_stderr_print(*args, **kwargs) -> None:
    """Best-effort human-readable output sink for ACP stdio sessions.

    ACP reserves stdout for JSON-RPC frames, so any incidental CLI/status output
    from AIAgent must be redirected away from stdout. Route it to stderr instead.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _register_task_cwd(task_id: str, cwd: str) -> None:
    """Bind a task/session id to the editor's working directory for tools.

    Zed can launch Hermes from a Windows workspace while the ACP process runs
    inside WSL. In that case ACP sends cwd as e.g. ``E:\\Projects\\POTI``;
    local tools need the WSL mount equivalent or subprocess creation fails
    before the command can run.
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides
        register_task_env_overrides(task_id, {"cwd": _translate_acp_cwd(cwd)})
    except Exception:
        logger.debug("Failed to register ACP task cwd override", exc_info=True)


def _expand_acp_enabled_toolsets(
    toolsets: List[str] | None = None,
    mcp_server_names: List[str] | None = None,
) -> List[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    expanded: List[str] = []
    for name in list(toolsets or ["hermes-acp"]):
        if name and name not in expanded:
            expanded.append(name)

    for server_name in list(mcp_server_names or []):
        toolset_name = f"mcp-{server_name}"
        if server_name and toolset_name not in expanded:
            expanded.append(toolset_name)

    return expanded


def _clear_task_cwd(task_id: str) -> None:
    """Remove task-specific cwd overrides for an ACP session."""
    if not task_id:
        return
    try:
        from tools.terminal_tool import clear_task_env_overrides
        clear_task_env_overrides(task_id)
    except Exception:
        logger.debug("Failed to clear ACP task cwd override", exc_info=True)


@dataclass
class SessionState:
    """Tracks per-session state for an ACP-managed Hermes agent."""

    session_id: str
    agent: Any  # AIAgent instance
    cwd: str = "."
    model: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = None  # threading.Event
    is_running: bool = False
    queued_prompts: List[str] = field(default_factory=list)
    runtime_lock: Any = field(default_factory=Lock)
    turn_terminal_lock: Any = field(default_factory=asyncio.Lock)
    turn_terminal_winner: str | None = None
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""
    transcript_correction_poisoned: bool = False
    transcript_correction_poison_persisted: bool | None = None


class OwnedSessions:
    """Tracks which ACP session ids THIS process may legitimately serve.

    Every hermes-acp process serving Switchboard can share one on-disk
    SessionDB (``~/.hermes/state.db``) with sibling processes, each serving
    an unrelated Switchboard/editor session (see ``get_hermes_home``).
    Without this gate, ``SessionManager.get_session`` happily restores ANY
    session id that happens to exist in that shared database — even one
    live in a DIFFERENT process's connection right now — letting output
    leak across sessions (#delegation-cross-session-leak, 2026-07-25).

    The first ``session/new`` or ``session/load``/``session/resume`` this
    process handles establishes its *primary* id (see :meth:`add` and
    :meth:`check_first_bind`). Every id this process creates afterwards —
    a fork, or any other in-process session creation — joins the owned set
    automatically. Every other ACP protocol entry point that takes a
    client-supplied ``session_id`` must be a member of the set or be
    refused; see the ``_guard_owned_session`` / ``_guard_first_bind``
    helpers on ``HermesACPAgent`` for where that refusal happens.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._owned: set[str] = set()
        self._primary_id: str | None = None

    @property
    def primary_id(self) -> str | None:
        with self._lock:
            return self._primary_id

    def is_owned(self, session_id: str) -> bool:
        """Return True if *session_id* belongs to this process's owned set."""
        if not session_id:
            return False
        with self._lock:
            return session_id in self._owned

    def add(self, session_id: str) -> None:
        """Record *session_id* as owned (in-process creation: new/fork/etc.).

        The first id ever added becomes the process's primary id. This is
        never refused — the id was minted or explicitly bound by this
        process itself, so there is nothing foreign to guard against.
        """
        if not session_id:
            return
        with self._lock:
            if self._primary_id is None:
                self._primary_id = session_id
            self._owned.add(session_id)

    def check_first_bind(self, session_id: str) -> str | None:
        """Enforce bind-on-load for ``session/load``/``session/resume``.

        Returns ``None`` when *session_id* is allowed — either it is
        already owned, or the bind is legitimate for this process's
        topology. Returns a human-readable denial reason otherwise.

        Two topologies exist. A *dedicated* process — spawned by Switchboard
        for exactly one conversation, marked by ``HERMES_SESSION_CHAT_ID``
        and/or ``HERMES_EXPECTED_ACP_SESSION_ID`` in its environment — may
        bind only once: its first ``session/load``/``session/resume`` (or
        ``session/new``) claims the process, and every later load of a
        different id is refused. A *generic* multi-session host (Zed, Buzz —
        one long-lived process serving several independent conversations)
        has neither marker set; each ``session/load`` of a not-yet-owned id
        is an additional legitimate bind. Handlers other than load/resume
        always require prior membership regardless of topology.
        """
        if not session_id:
            return "empty session id"
        expected = (os.environ.get("HERMES_EXPECTED_ACP_SESSION_ID") or "").strip()
        dedicated = bool(
            expected or (os.environ.get("HERMES_SESSION_CHAT_ID") or "").strip()
        )
        with self._lock:
            if session_id in self._owned:
                return None
            if self._owned:
                if dedicated:
                    return "not owned by this process"
                self._owned.add(session_id)
                return None
            if expected and expected != session_id:
                return f"does not match spawn-pinned session {expected!r}"
            if self._primary_id is None:
                self._primary_id = session_id
            self._owned.add(session_id)
            return None


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the
    shared SessionDB so they survive process restarts and are searchable
    via ``session_search``.
    """

    def __init__(self, agent_factory=None, db=None):
        """
        Args:
            agent_factory: Optional callable that creates an AIAgent-like object.
                           Used by tests. When omitted, a real AIAgent is created
                           using the current Hermes runtime provider configuration.
            db:            Optional SessionDB instance. When omitted, the default
                           SessionDB (``~/.hermes/state.db``) is lazily created.
        """
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use
        self._owned_sessions = OwnedSessions()

    @property
    def owned_sessions(self) -> OwnedSessions:
        """This process's session-ownership gate (see :class:`OwnedSessions`)."""
        return self._owned_sessions

    # ---- public API ---------------------------------------------------------

    def create_session(self, cwd: str = ".") -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent."""
        import threading

        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=session_id, cwd=cwd)
        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", "") or "",
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[session_id] = state
        self._owned_sessions.add(session_id)
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the session for *session_id*, or ``None``.

        If the session is not in memory but exists in the database (e.g. after
        a process restart), it is transparently restored.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            self._raise_if_transcript_poisoned(state)
            return state
        # Attempt to restore from database.
        return self._restore(session_id)

    def peek_session(self, session_id: str) -> Optional[SessionState]:
        """Return *session_id* only if it is already resident in THIS process.

        Unlike :meth:`get_session`, this never falls back to restoring from
        the (potentially shared, multi-process) SessionDB. Deliberately
        stricter for callers that must prove **live ownership** of a session
        before acting on its behalf — e.g. routing a background-delegation
        completion — where "exists in the database" only means some process,
        possibly a different one working an unrelated conversation, created
        that session at some point. It does NOT mean this process is that
        session's current, legitimate owner. When multiple ACP processes
        share one SessionDB (e.g. a host that doesn't isolate HERMES_HOME
        per process), ``get_session`` would silently adopt and mutate a
        session this process was never asked to load or resume, and any
        outbound ACP notification would then be sent over the WRONG
        connection under a foreign session_id (#delegation-cross-session-leak).
        """
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from memory and database. Returns True if it existed."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        db_existed = self._delete_persisted(session_id)
        if existed or db_existed:
            _clear_task_cwd(session_id)
        return existed or db_existed

    def fork_session(self, session_id: str, cwd: str = ".") -> Optional[SessionState]:
        """Deep-copy a session's history into a new session."""
        import threading

        from acp_adapter.orchestration import requested_orchestration_mode

        if requested_orchestration_mode() is not None:
            raise RuntimeError(
                "ACP session/fork is unavailable for a Switchboard-managed "
                "orchestration session; a fork requires a distinct gateway "
                "record and credential"
            )

        cwd = _translate_acp_cwd(cwd)
        original = self.get_session(session_id)  # checks DB too
        if original is None:
            return None

        new_id = str(uuid.uuid4())
        agent = self._make_agent(
            session_id=new_id,
            cwd=cwd,
            model=original.model or None,
        )
        state = SessionState(
            session_id=new_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", original.model) or original.model,
            history=copy.deepcopy(original.history),
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[new_id] = state
        self._owned_sessions.add(new_id)
        _register_task_cwd(new_id, cwd)
        self._persist(state)
        logger.info("Forked ACP session %s -> %s", session_id, new_id)
        return state

    def list_sessions(self, cwd: str | None = None) -> List[Dict[str, Any]]:
        """Return lightweight info dicts for all sessions (memory + database)."""
        normalized_cwd = _normalize_cwd_for_compare(cwd) if cwd else None
        db = self._get_db()
        persisted_rows: dict[str, dict[str, Any]] = {}

        if db is not None:
            try:
                for row in db.list_sessions_rich(source="acp", limit=1000):
                    persisted_rows[str(row["id"])] = dict(row)
            except Exception:
                logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        # Collect in-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                history_len = len(s.history)
                if history_len <= 0:
                    continue
                if normalized_cwd and _normalize_cwd_for_compare(s.cwd) != normalized_cwd:
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                preview = next(
                    (
                        str(msg.get("content") or "").strip()
                        for msg in s.history
                        if msg.get("role") == "user" and str(msg.get("content") or "").strip()
                    ),
                    persisted.get("preview") or "",
                )
                results.append(
                    {
                        "session_id": s.session_id,
                        "cwd": s.cwd,
                        "model": s.model,
                        "history_len": history_len,
                        "title": _build_session_title(persisted.get("title"), preview, s.cwd),
                        "updated_at": _format_updated_at(
                            persisted.get("last_active") or persisted.get("started_at") or time.time()
                        ),
                    }
                )

        # Merge any persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            if sid in seen_ids:
                continue
            message_count = int(row.get("message_count") or 0)
            if message_count <= 0:
                continue
            # Extract cwd from model_config JSON.
            session_cwd = "."
            mc = row.get("model_config")
            if mc:
                try:
                    session_cwd = json.loads(mc).get("cwd", ".")
                except (json.JSONDecodeError, TypeError):
                    pass
            if normalized_cwd and _normalize_cwd_for_compare(session_cwd) != normalized_cwd:
                continue
            results.append({
                "session_id": sid,
                "cwd": session_cwd,
                "model": row.get("model") or "",
                "history_len": message_count,
                "title": _build_session_title(row.get("title"), row.get("preview"), session_cwd),
                "updated_at": _format_updated_at(row.get("last_active") or row.get("started_at")),
            })

        results.sort(key=lambda item: _updated_at_sort_key(item.get("updated_at")), reverse=True)
        return results

    def update_cwd(self, session_id: str, cwd: str) -> Optional[SessionState]:
        """Update the working directory for a session and its tool overrides."""
        cwd = _translate_acp_cwd(cwd)
        state = self.get_session(session_id)  # checks DB too
        if state is None:
            return None
        state.cwd = cwd
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        return state

    def cleanup(self) -> None:
        """Remove all sessions (memory and database) and clear task-specific cwd overrides."""
        with self._lock:
            session_ids = list(self._sessions.keys())
            self._sessions.clear()
        for session_id in session_ids:
            _clear_task_cwd(session_id)
            self._delete_persisted(session_id)
        # Also remove any DB-only ACP sessions not currently in memory.
        db = self._get_db()
        if db is not None:
            try:
                rows = db.search_sessions(source="acp", limit=10000)
                for row in rows:
                    sid = row["id"]
                    _clear_task_cwd(sid)
                    db.delete_session(sid)
            except Exception:
                logger.debug("Failed to cleanup ACP sessions from DB", exc_info=True)

    def save_session(self, session_id: str) -> bool:
        """Persist the current state of a session to the database.

        Called by the server after prompt completion, slash commands that
        mutate history, and model switches.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return self._persist(state)
        return False

    def mark_transcript_correction_poisoned(self, state: SessionState) -> bool:
        """Persist a fail-closed marker without rewriting unsafe messages."""
        state.transcript_correction_poisoned = True
        persisted = self._persist(state, persist_history=False)
        state.transcript_correction_poison_persisted = persisted
        return persisted

    def clear_transcript_correction_poisoned(self, state: SessionState) -> bool:
        """Clear the marker only after the caller verifies durable correction."""
        state.transcript_correction_poisoned = False
        persisted = self._persist(state, persist_history=False)
        if not persisted:
            # The durable row may still be poisoned. Keep the in-memory owner
            # blocked too rather than claiming the session is safe.
            state.transcript_correction_poisoned = True
            state.transcript_correction_poison_persisted = False
            return False
        state.transcript_correction_poison_persisted = None
        return True

    # ---- persistence via SessionDB ------------------------------------------

    def _get_db(self):
        """Lazily initialise and return the SessionDB instance.

        Returns ``None`` if the DB is unavailable (e.g. import error in a
        minimal test environment).

        Note: we resolve ``HERMES_HOME`` dynamically rather than relying on
        the module-level ``DEFAULT_DB_PATH`` constant, because that constant
        is evaluated at import time and won't reflect env-var changes made
        later (e.g. by the test fixture ``_isolate_hermes_home``).
        """
        if self._db_instance is not None:
            return self._db_instance
        try:
            from hermes_state import SessionDB
            hermes_home = get_hermes_home()
            self._db_instance = SessionDB(db_path=hermes_home / "state.db")
            return self._db_instance
        except Exception:
            logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
            return None

    def _persist(
        self,
        state: SessionState,
        *,
        persist_history: bool = True,
    ) -> bool:
        """Write session state to the database.

        Creates the session record if it doesn't exist, then replaces all
        stored messages with the current in-memory history.
        """
        db = self._get_db()
        if db is None:
            return True

        # Ensure model is a plain string (not a MagicMock or other proxy).
        model_str = str(state.model) if state.model else None
        session_meta = {"cwd": state.cwd}
        provider = getattr(state.agent, "provider", None)
        base_url = getattr(state.agent, "base_url", None)
        api_mode = getattr(state.agent, "api_mode", None)
        if isinstance(provider, str) and provider.strip():
            session_meta["provider"] = provider.strip()
        if isinstance(base_url, str) and base_url.strip():
            session_meta["base_url"] = base_url.strip()
        if isinstance(api_mode, str) and api_mode.strip():
            session_meta["api_mode"] = api_mode.strip()
        if state.transcript_correction_poisoned:
            session_meta[_TRANSCRIPT_CORRECTION_POISON_KEY] = True
        cwd_json = json.dumps(session_meta)

        try:
            # Ensure the session record exists.
            existing = db.get_session(state.session_id)
            if existing is None:
                db.create_session(
                    session_id=state.session_id,
                    source="acp",
                    model=model_str,
                    model_config=session_meta,
                )
            else:
                # Update model_config (contains cwd) if changed.
                db.update_session_meta(state.session_id, cwd_json, model_str)

            if not persist_history:
                return True

            # When the agent owns persistence to this same SessionDB it has
            # already flushed the live transcript incrementally during
            # run_conversation (append_message), and it preserves pre-compaction
            # turns non-destructively via archive_and_compact() — keeping them on
            # disk as searchable active=0/compacted=1 rows. Calling
            # replace_messages() here would then be a redundant double-write that
            # DELETEs exactly those archived rows (and, after a compression-driven
            # id rotation where agent.session_id no longer equals
            # state.session_id, clobbers the ended parent transcript) — silent
            # data loss for any ACP conversation long enough to compress.
            #
            # Only fall back to the destructive atomic replace when the agent is
            # NOT persisting itself to this DB (e.g. a test agent factory, or a
            # fresh create/fork whose copied history the agent has not flushed
            # yet). That path still rolls back on a mid-rewrite failure so the
            # previously persisted conversation survives (salvaged from #13675).
            agent = state.agent
            agent_db = getattr(agent, "_session_db", None)
            agent_owns_persistence = (
                agent_db is not None
                and agent_db is db
                and bool(getattr(agent, "_session_db_created", False))
            )
            if not agent_owns_persistence:
                # Even when the current agent doesn't "own" persistence, the
                # session on disk may already carry compaction-archived rows —
                # e.g. after a model switch or a /restore, both of which mint a
                # fresh agent with _session_db_created=False (so the check above
                # is False) yet leave the durable archived transcript in place.
                # A full-history replace would DELETE those archived rows just
                # like the owned-agent case. Guard against it by replacing ONLY
                # the live (active=1) set unconditionally: on a fresh
                # create/fork every row is active=1, so active-only replace is
                # behaviorally identical to the full replace — and when archived
                # rows DO exist they survive. An existence probe here
                # (has_archived_messages) would fail OPEN into the destructive
                # replace on any DB error and can race a concurrent
                # archive_and_compact — the same probe failure mode #80216's
                # /retry fix (gateway/slash_commands.py) deliberately avoids.
                db.replace_messages(
                    state.session_id, state.history, active_only=True
                )
            return True
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)
            return False

    def _self_heal_poisoned_history(
        self,
        *,
        db: Any,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Attempt to redo a stuck correction on a poisoned session at restore time.

        The durable poison marker means "correction may not have completed" —
        it is a redo flag, not a tombstone. Two shapes reach here:

        1. ``replace_messages`` genuinely succeeded last time and only the
           subsequent marker CLEAR failed transiently (the round-2 defect
           this method exists to fix): ``history`` is already the sanitized,
           safe version. Re-running sanitize is then a true no-op (nothing
           left to strip — ``_sanitize_required_assistant_candidate`` clears
           an already-empty ``content``/absent ``api_content``/etc
           idempotently), re-running ``replace_messages`` rewrites the same
           safe bytes, and only the clear needs to actually take effect this
           time.
        2. The correction never completed at all (a genuine crash-window
           hit, or a persistently failing correction): ``history`` still
           carries the tainted candidate. Sanitize strips it for real this
           time, and the rewrite durably persists the correction.

        In both cases the boundary is "everything after the most recent
        user message" — the only turn a poisoned session can ever have
        pending, since a poisoned session can never accept a new user turn
        (``_raise_if_transcript_poisoned`` / this same poison check refuses
        before one could ever be appended). Forcing
        ``_sanitize_failed_turn_history``'s baseline-clamp fallback (passing
        a ``baseline_count`` past the end of the list) computes exactly that
        boundary instead of risking sanitizing an unrelated, already-
        completed earlier turn.

        Returns the (possibly sanitized) history on success, or ``None`` if
        the redo could not be proven safe — callers must then refuse exactly
        as before (fail closed, unchanged from the pre-self-heal behavior).
        """
        from acp_adapter.server import (
            _rewrite_agent_active_history,
            _sanitize_failed_turn_history,
        )

        safe_history = _sanitize_failed_turn_history(
            history, baseline_count=len(history) + 1
        )
        correction_target = SimpleNamespace(_session_db=db, session_id=session_id)
        temp_state = SessionState(
            session_id=session_id,
            agent=correction_target,
            transcript_correction_poisoned=True,
        )
        try:
            healed = _rewrite_agent_active_history(
                correction_target, safe_history, temp_state, self
            )
        except Exception:
            logger.warning(
                "Poisoned-session self-heal redo raised for ACP session %s",
                session_id,
                exc_info=True,
            )
            return None
        if not healed:
            logger.warning(
                "Poisoned-session self-heal redo did not complete for ACP "
                "session %s; refusing resume",
                session_id,
            )
            return None
        logger.info(
            "Poisoned ACP session %s self-healed on restore (%d messages)",
            session_id,
            len(safe_history),
        )
        return safe_history

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load a session from the database into memory, recreating the AIAgent."""
        import threading

        db = self._get_db()
        if db is None:
            return None

        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for ACP session %s", session_id, exc_info=True)
            return None

        if row is None:
            return None

        # Only restore ACP sessions.
        if row.get("source") != "acp":
            return None

        # Extract cwd/provider metadata from model_config. Reading this is
        # safe even for a poisoned session — it is routing/identity data,
        # not the protected message content the poison marker guards.
        cwd = "."
        requested_provider = row.get("billing_provider")
        restored_base_url = row.get("billing_base_url")
        restored_api_mode = None
        poisoned = False
        mc = row.get("model_config")
        if mc:
            try:
                meta = json.loads(mc)
                if isinstance(meta, dict):
                    poisoned = bool(meta.get(_TRANSCRIPT_CORRECTION_POISON_KEY))
                    cwd = meta.get("cwd", ".")
                    requested_provider = meta.get("provider") or requested_provider
                    restored_base_url = meta.get("base_url") or restored_base_url
                    restored_api_mode = meta.get("api_mode") or restored_api_mode
            except (json.JSONDecodeError, TypeError):
                pass

        model = row.get("model") or None

        # Load conversation history. repair_alternation: this restore feeds
        # LIVE REPLAY — the loaded list becomes the resumed agent's working
        # conversation. A durable ``user;user`` violation left in state.db would
        # otherwise re-fire the pre-request defensive repair on every request
        # for the rest of the session (see hermes_state.get_messages_as_conversation).
        try:
            history = db.get_messages_as_conversation(
                session_id, repair_alternation=True
            )
        except Exception:
            logger.warning("Failed to load messages for ACP session %s", session_id, exc_info=True)
            history = []

        if poisoned:
            # The marker means "correction may not have completed" — redo it
            # before resuming. Proceed only on full success; otherwise keep
            # refusing exactly as before self-heal existed.
            healed_history = self._self_heal_poisoned_history(
                db=db, session_id=session_id, history=history
            )
            if healed_history is None:
                raise UnsafeSessionTranscriptError(session_id)
            history = healed_history

        try:
            agent = self._make_agent(
                session_id=session_id,
                cwd=cwd,
                model=model,
                requested_provider=requested_provider,
                base_url=restored_base_url,
                api_mode=restored_api_mode,
            )
        except Exception:
            logger.warning("Failed to recreate agent for ACP session %s", session_id, exc_info=True)
            return None

        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=model or getattr(agent, "model", "") or "",
            history=history,
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        logger.info("Restored ACP session %s from DB (%d messages)", session_id, len(history))
        return state

    @staticmethod
    def _raise_if_transcript_poisoned(state: SessionState) -> None:
        if state.transcript_correction_poisoned:
            raise UnsafeSessionTranscriptError(state.session_id)

    def _delete_persisted(self, session_id: str) -> bool:
        """Delete a session from the database. Returns True if it existed."""
        db = self._get_db()
        if db is None:
            return False
        try:
            return db.delete_session(session_id)
        except Exception:
            logger.debug("Failed to delete ACP session %s from DB", session_id, exc_info=True)
            return False

    # ---- internal -----------------------------------------------------------

    def _make_agent(
        self,
        *,
        session_id: str,
        cwd: str,
        model: str | None = None,
        requested_provider: str | None = None,
        base_url: str | None = None,
        api_mode: str | None = None,
    ):
        if self._agent_factory is not None:
            return self._agent_factory()

        from run_agent import AIAgent
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_constants import parse_reasoning_effort

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or default_model)
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        from acp_adapter.orchestration import without_reserved_switchboard_mcp

        configured_mcp_servers = [
            name
            for name, cfg in without_reserved_switchboard_mcp(
                config.get("mcp_servers")
            ).items()
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is not False
        ]
        agent_cfg = config.get("agent")
        configured_reasoning_effort = (
            str(agent_cfg.get("reasoning_effort") or "")
            if isinstance(agent_cfg, dict)
            else ""
        )
        session_reasoning_effort = (
            os.environ.get("HERMES_SESSION_REASONING_EFFORT") or ""
        ).strip().lower()
        requested_reasoning_effort = (
            session_reasoning_effort or configured_reasoning_effort
        ).strip().lower()
        reasoning_config = (
            {"enabled": True, "effort": requested_reasoning_effort}
            if requested_reasoning_effort in {"max", "ultra"}
            else parse_reasoning_effort(requested_reasoning_effort)
        )
        effective_reasoning_effort = (
            str(reasoning_config.get("effort") or "").lower()
            if isinstance(reasoning_config, dict)
            else ""
        )

        kwargs = {
            "platform": "acp",
            "enabled_toolsets": _expand_acp_enabled_toolsets(
                ["hermes-acp"],
                mcp_server_names=configured_mcp_servers,
            ),
            "quiet_mode": True,
            "session_id": session_id,
            "session_db": self._get_db(),
            "model": model or default_model,
            "reasoning_config": reasoning_config,
        }

        # Internal Switchboard bridge contract: apply the session-scoped
        # orchestration owner before AIAgent snapshots its tools. Invalid or
        # incomplete enforcement fails construction rather than silently
        # exposing Hermes native delegation in Single/Switchboard mode.
        from acp_adapter.orchestration import apply_orchestration_tool_policy

        apply_orchestration_tool_policy(kwargs)

        try:
            runtime = resolve_runtime_provider(requested=requested_provider or config_provider)
            runtime_provider = runtime.get("provider")
            runtime_api_mode = api_mode or runtime.get("api_mode")
            if (
                effective_reasoning_effort in {"max", "ultra"}
                and runtime_provider == "openai-codex"
            ):
                runtime_api_mode = "codex_app_server"
            kwargs.update(
                {
                    "provider": runtime_provider,
                    "api_mode": runtime_api_mode,
                    "base_url": base_url or runtime.get("base_url"),
                    "api_key": runtime.get("api_key"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                }
            )
        except Exception:
            logger.debug("ACP session falling back to default provider resolution", exc_info=True)

        _register_task_cwd(session_id, cwd)

        # Bounded wait for background MCP discovery so already-spawning fast
        # servers land in the agent's tool snapshot.  ACP entry.py fires
        # discovery in a background daemon thread (start_background_mcp_discovery);
        # the agent snapshots tools once at build (run_agent/agent_init) and
        # never re-reads the registry, so without this join a reachable-but-
        # slow configured server would be invisible for the whole session.
        # ``ensure_mcp_discovery_before_agent_build`` also (re)starts discovery
        # when the entry.py spawn never ran or exited with zero connected
        # servers (the retry-after-zero-connected allowance), making this
        # construction site self-sufficient.  Bounded by
        # ``mcp_discovery_timeout`` (config.yaml, default ~1.5s) so a dead
        # server can't block — servers that miss the bound are picked up by
        # the automatic late-refresh (see HermesACPAgent._schedule_mcp_late_refresh).
        try:
            from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build

            ensure_mcp_discovery_before_agent_build(
                logger=logger,
                thread_name="acp-mcp-discovery",
            )
        except Exception:
            logger.debug("ACP: bounded MCP discovery wait failed", exc_info=True)

        agent = AIAgent(**kwargs)
        # Codex app-server sessions are spawned lazily on the first turn. Stamp
        # the ACP workspace onto the agent so the Codex runtime starts from the
        # editor/session cwd instead of the Hermes daemon's process cwd.
        agent.session_cwd = cwd
        # ACP stdio transport requires stdout to remain protocol-only JSON-RPC.
        # Route any incidental human-readable agent output to stderr instead.
        agent._print_fn = _acp_stderr_print
        return agent
