"""Per-process session ownership gate at the ACP protocol boundary.

Every hermes-acp process can share one on-disk SessionDB
(``~/.hermes/state.db``) with sibling processes, each serving an unrelated
Switchboard/editor session (see ``get_hermes_home``). Before this gate,
``SessionManager.get_session`` happily restored ANY session id that
happened to exist in that shared database — even one live in a DIFFERENT
process's connection right now — letting output leak across sessions
(#delegation-cross-session-leak, 2026-07-25).

This suite locks down:
  - a bound process refuses a foreign session id at every ACP protocol
    handler that takes a client-supplied ``session_id``
  - fork/child ids created in-process are always allowed, including as the
    target of a follow-up ``session/prompt``
  - ``HERMES_EXPECTED_ACP_SESSION_ID`` spawn-time pinning is enforced on the
    first bind (``session/load`` / ``session/resume``)
  - a legitimate relaunch — a fresh process's first ``session/load`` or
    ``session/resume`` against an existing, previously-unbound id — still
    works, since that is exactly how the gateway reconnects a session
"""

from types import SimpleNamespace

import pytest
from acp.exceptions import RequestError
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import OwnedSessions, SessionManager
from tools.process_registry import process_registry


class _FakeAgent:
    """Minimal AIAgent stand-in — enough surface for prompt()/cancel()."""

    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.runs = []
        self._required_delegation_launching = False

    def _has_unconsumed_required_delegations(self):
        return False

    def _finish_acp_provisional_stream(self, *, discard):
        return None

    def run_conversation(self, *, user_message, conversation_history, task_id, **_kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}

    def interrupt(self):
        return None


class _CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))

    async def request_permission(self, *_args, **_kwargs):
        return SimpleNamespace(outcome="allow")


class _NoopDb:
    """Same minimal shape already proven safe across this test suite."""

    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None

    def replace_messages(self, *_args, **_kwargs):
        return None

    def has_archived_messages(self, *_args, **_kwargs):
        return False


class _FakeDbWithRow:
    """Minimal DB stub with exactly one persisted ACP session row.

    Models the shared SessionDB a fresh (relaunched) process would see: the
    row already exists, written by some earlier process/turn.
    """

    def __init__(self, session_id, *, model_config=None):
        self._session_id = session_id
        self._row = {
            "source": "acp",
            "model_config": model_config,
            "billing_provider": None,
            "billing_base_url": None,
            "model": "fake-model",
        }

    def get_session(self, session_id):
        if session_id == self._session_id:
            return dict(self._row)
        return None

    def get_messages_as_conversation(self, session_id, repair_alternation=True):
        return []

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session_meta(self, *_args, **_kwargs):
        return None

    def replace_messages(self, *_args, **_kwargs):
        return None

    def has_archived_messages(self, *_args, **_kwargs):
        return False


@pytest.fixture(autouse=True)
def _clear_pin_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXPECTED_ACP_SESSION_ID", raising=False)


@pytest.fixture(autouse=True)
def _clean_completion_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_prompt_agent(monkeypatch, *, db=None):
    fake = _FakeAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=db or _NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
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
    return acp_agent, manager, fake, conn


# ---------------------------------------------------------------------------
# OwnedSessions — direct unit coverage of the tracking primitive
# ---------------------------------------------------------------------------


def test_owned_sessions_starts_empty_and_unbound():
    owned = OwnedSessions()
    assert owned.primary_id is None
    assert owned.is_owned("anything") is False


def test_owned_sessions_add_binds_first_id_and_accumulates():
    owned = OwnedSessions()
    owned.add("primary-id")
    assert owned.primary_id == "primary-id"
    assert owned.is_owned("primary-id")
    assert not owned.is_owned("other-id")

    owned.add("child-id")
    assert owned.primary_id == "primary-id"  # unchanged by later additions
    assert owned.is_owned("child-id")


def test_owned_sessions_check_first_bind_allows_unpinned_first_call(monkeypatch):
    monkeypatch.delenv("HERMES_EXPECTED_ACP_SESSION_ID", raising=False)
    owned = OwnedSessions()
    assert owned.check_first_bind("some-id") is None
    assert owned.is_owned("some-id")
    assert owned.primary_id == "some-id"


def test_owned_sessions_check_first_bind_refuses_foreign_id_once_bound():
    owned = OwnedSessions()
    owned.add("primary-id")
    denial = owned.check_first_bind("foreign-id")
    assert denial is not None
    assert not owned.is_owned("foreign-id")


def test_owned_sessions_check_first_bind_pin_mismatch_refused(monkeypatch):
    monkeypatch.setenv("HERMES_EXPECTED_ACP_SESSION_ID", "expected-id")
    owned = OwnedSessions()
    denial = owned.check_first_bind("some-other-id")
    assert denial is not None
    assert owned.primary_id is None
    assert not owned.is_owned("some-other-id")


def test_owned_sessions_check_first_bind_pin_match_allowed(monkeypatch):
    monkeypatch.setenv("HERMES_EXPECTED_ACP_SESSION_ID", "expected-id")
    owned = OwnedSessions()
    denial = owned.check_first_bind("expected-id")
    assert denial is None
    assert owned.is_owned("expected-id")
    assert owned.primary_id == "expected-id"


# ---------------------------------------------------------------------------
# (e) unbound first session/new is always allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_session_new_is_always_allowed_and_becomes_owned(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)

    response = await acp_agent.new_session(cwd=".")

    assert manager.owned_sessions.is_owned(response.session_id)
    assert manager.owned_sessions.primary_id == response.session_id


# ---------------------------------------------------------------------------
# (a) foreign prompt refused after bind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_for_foreign_session_is_refused_after_bind(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    owned_state = manager.create_session(cwd=".")

    with pytest.raises(RequestError) as exc_info:
        await acp_agent.prompt(
            session_id="someone-elses-session",
            prompt=[TextContentBlock(type="text", text="hello")],
        )

    assert "someone-elses-session" in str(exc_info.value)
    assert manager.owned_sessions.is_owned(owned_state.session_id)
    assert not manager.owned_sessions.is_owned("someone-elses-session")


@pytest.mark.asyncio
async def test_prompt_for_owned_session_still_works_after_bind(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    state = manager.create_session(cwd=".")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="hello")],
    )

    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_cancel_for_foreign_session_is_refused_after_bind(monkeypatch, caplog):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with caplog.at_level("WARNING", logger="acp_adapter.server"):
        with pytest.raises(RequestError):
            await acp_agent.cancel("someone-elses-session")

    guard_lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("cross-session guard:")
    ]
    assert len(guard_lines) == 1
    assert "session/cancel" in guard_lines[0]
    assert "someone-elses-session" in guard_lines[0]


# ---------------------------------------------------------------------------
# (b) foreign load_session refused after bind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_session_for_foreign_id_is_refused_after_bind(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.load_session(cwd=".", session_id="someone-elses-session")


@pytest.mark.asyncio
async def test_resume_session_for_foreign_id_is_refused_after_bind(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.resume_session(cwd=".", session_id="someone-elses-session")


@pytest.mark.asyncio
async def test_load_session_as_first_bind_on_fresh_process_restores_existing_session(
    monkeypatch,
):
    """Legitimate relaunch: a brand-new process's first session/load against
    a pre-existing (not-yet-owned-by-this-process) id must still work — this
    is how the gateway reconnects a resumed conversation.
    """
    existing_id = "existing-remote-session-id"
    db = _FakeDbWithRow(existing_id)
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch, db=db)

    response = await acp_agent.load_session(cwd=".", session_id=existing_id)

    assert response is not None
    assert manager.owned_sessions.is_owned(existing_id)
    assert manager.owned_sessions.primary_id == existing_id


@pytest.mark.asyncio
async def test_load_session_for_second_different_id_refused_once_bound(monkeypatch):
    """The FIRST session/load binds the process; a second, different id is
    then just another foreign reference and must be refused like any other
    protocol handler.
    """
    first_id = "first-loaded-session"
    db = _FakeDbWithRow(first_id)
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch, db=db)

    first = await acp_agent.load_session(cwd=".", session_id=first_id)
    assert first is not None

    with pytest.raises(RequestError):
        await acp_agent.load_session(cwd=".", session_id="a-totally-different-session")


# ---------------------------------------------------------------------------
# (c) fork child id allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_child_id_allowed_and_usable_for_a_follow_up_prompt(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    parent = manager.create_session(cwd=".")

    fork_response = await acp_agent.fork_session(cwd=".", session_id=parent.session_id)

    assert fork_response.session_id
    assert fork_response.session_id != parent.session_id
    assert manager.owned_sessions.is_owned(fork_response.session_id)

    prompt_response = await acp_agent.prompt(
        session_id=fork_response.session_id,
        prompt=[TextContentBlock(type="text", text="continue")],
    )
    assert prompt_response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_fork_session_refused_for_foreign_parent_id(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.fork_session(cwd=".", session_id="someone-elses-session")


# ---------------------------------------------------------------------------
# (d) HERMES_EXPECTED_ACP_SESSION_ID mismatch refused / match allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expected_session_id_pin_refuses_mismatched_first_load(monkeypatch):
    monkeypatch.setenv("HERMES_EXPECTED_ACP_SESSION_ID", "expected-session-id")
    db = _FakeDbWithRow("actually-loaded-id")
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch, db=db)

    with pytest.raises(RequestError):
        await acp_agent.load_session(cwd=".", session_id="actually-loaded-id")

    assert not manager.owned_sessions.is_owned("actually-loaded-id")
    assert manager.owned_sessions.primary_id is None


@pytest.mark.asyncio
async def test_expected_session_id_pin_allows_matching_first_load(monkeypatch):
    expected_id = "expected-session-id"
    monkeypatch.setenv("HERMES_EXPECTED_ACP_SESSION_ID", expected_id)
    db = _FakeDbWithRow(expected_id)
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch, db=db)

    response = await acp_agent.load_session(cwd=".", session_id=expected_id)

    assert response is not None
    assert manager.owned_sessions.is_owned(expected_id)
    assert manager.owned_sessions.primary_id == expected_id


@pytest.mark.asyncio
async def test_expected_session_id_pin_refuses_mismatched_first_resume(monkeypatch):
    monkeypatch.setenv("HERMES_EXPECTED_ACP_SESSION_ID", "expected-session-id")
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)

    with pytest.raises(RequestError):
        await acp_agent.resume_session(cwd=".", session_id="someone-elses-session")

    assert manager.owned_sessions.primary_id is None


# ---------------------------------------------------------------------------
# Model/mode/config-option protocol handlers also enforce ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_session_model_refused_for_foreign_session(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.set_session_model(model_id="whatever", session_id="foreign-id")


@pytest.mark.asyncio
async def test_set_session_mode_refused_for_foreign_session(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.set_session_mode(mode_id="default", session_id="foreign-id")


@pytest.mark.asyncio
async def test_set_config_option_refused_for_foreign_session(monkeypatch):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    manager.create_session(cwd=".")

    with pytest.raises(RequestError):
        await acp_agent.set_config_option(
            config_id="edit_approval_policy",
            session_id="foreign-id",
            value="ask",
        )


# ---------------------------------------------------------------------------
# Log line format — must be grep-able as "cross-session guard: ..."
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_log_line_matches_required_format(monkeypatch, caplog):
    acp_agent, manager, _fake, _conn = _make_prompt_agent(monkeypatch)
    owned_state = manager.create_session(cwd=".")

    with caplog.at_level("WARNING", logger="acp_adapter.server"):
        with pytest.raises(RequestError):
            await acp_agent.prompt(
                session_id="foreign-id",
                prompt=[TextContentBlock(type="text", text="hi")],
            )

    matching = [
        r for r in caplog.records if r.getMessage().startswith("cross-session guard:")
    ]
    assert len(matching) == 1
    message = matching[0].getMessage()
    assert "session/prompt" in message
    assert "foreign-id" in message
    assert owned_state.session_id in message
