"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.model_catalog import ACP_MAX_MODELS_PER_PROVIDER
from acp_adapter.server import (
    HermesACPAgent,
    HERMES_VERSION,
)
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    def _agent_factory():
        agent = MagicMock(name="MockAIAgent")
        agent._required_delegation_launching = False
        agent._has_unconsumed_required_delegations.return_value = False
        return agent

    return SessionManager(agent_factory=_agent_factory)


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [(mode.id, mode.name) for mode in resp.modes.available_modes] == [
        ("default", "Default"),
        ("accept_edits", "Accept Edits"),
        ("dont_ask", "Don't Ask"),
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION




    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build_payload,
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        assert [model.name for model in resp.models.available_models] == [
            "Anthropic · claude-sonnet-4-6",
            "OpenAI Codex · gpt-5.4",
            "OpenAI Codex · gpt-5.4-mini",
        ]
        assert resp.models.available_models[1].description is not None
        assert "current" in resp.models.available_models[1].description
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )
        build_payload.assert_called_once_with(
            picker_context,
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



    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None

    def test_advertised_commands_match_handlers(self, agent):
        """Every advertised command is exactly the executable set the mixin dispatches."""
        advertised_names = {cmd.name for cmd in agent._available_commands()}
        assert advertised_names == set(type(agent)._COMMANDS)
        for name in advertised_names:
            assert callable(getattr(agent, f"_cmd_{name}", None))

    def test_advertised_commands_come_from_registry(self, agent):
        """Advertised description/hint are derived from hermes_cli's CommandDef registry,
        except entries explicitly listed in ACP_COMMAND_OVERRIDES (registry wording that
        doesn't hold for ACP's simpler handlers)."""
        from hermes_cli.commands import resolve_command
        from acp_adapter.commands import ACP_COMMAND_OVERRIDES

        # "skill" has no hermes_cli.commands registry entry at all: the CLI/gateway only register
        # a *management* command for skills as a whole ("skills"), never a static CommandDef for
        # loading one skill by name (that's a dynamically-registered "/<skill-name>" command per
        # agent/skill_commands.py). Every other ACP_COMMAND_OVERRIDES entry still wraps a real
        # registry command with different wording — only "skill" has nothing to derive from.
        REGISTRY_LESS_OVERRIDES = {"skill"}

        for cmd in agent._available_commands():
            registry_def = resolve_command(cmd.name)
            if cmd.name in REGISTRY_LESS_OVERRIDES:
                assert registry_def is None, f"/{cmd.name} unexpectedly has a registry entry now"
                continue
            assert registry_def is not None, f"/{cmd.name} has no hermes_cli.commands registry entry"

            if cmd.name in ACP_COMMAND_OVERRIDES:
                continue

            assert cmd.description == registry_def.description
            expected_hint = registry_def.args_hint.strip("[]<>").strip() or None
            if expected_hint:
                assert cmd.input is not None
                assert cmd.input.root.hint == expected_hint
            else:
                assert cmd.input is None


    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result["configOptions"] == []




    @pytest.mark.asyncio
    async def test_set_session_model_routes_switchboard_terra_ultra_to_app_server(
        self, tmp_path, monkeypatch
    ):
        def fake_resolve_runtime_provider(requested=None, **kwargs):
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "codex_responses" if provider == "openai-codex" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
                reasoning_config=kwargs.get("reasoning_config"),
            )

        monkeypatch.setenv("HERMES_SESSION_REASONING_EFFORT", "ultra")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {"provider": "openai-codex", "default": "gpt-5.5"},
                "mcp_servers": {},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = await acp_agent.set_session_model(
                model_id="openai-codex:gpt-5.6-terra",
                session_id=state.session_id,
            )

        assert isinstance(result, SetSessionModelResponse)
        assert state.model == "gpt-5.6-terra"
        assert state.agent.provider == "openai-codex"
        assert state.agent.api_mode == "codex_app_server"
        assert state.agent.reasoning_config == {"enabled": True, "effort": "ultra"}

    def test_explicit_current_provider_skips_auto_detection(self):
        with patch(
            "hermes_cli.models.detect_provider_for_model",
            side_effect=AssertionError("explicit provider must not be auto-detected"),
        ):
            selection = HermesACPAgent._resolve_model_selection(
                "openai-codex:gpt-5.6-terra",
                "openai-codex",
            )

        assert selection == ("openai-codex", "gpt-5.6-terra")

    @pytest.mark.asyncio
    async def test_failed_model_rebuild_keeps_previous_agent_and_model(self):
        previous_agent = SimpleNamespace(
            model="gpt-5.5",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
        )
        manager = SessionManager(agent_factory=lambda: previous_agent, db=None)
        acp_agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd="/tmp")

        def fail_agent_rebuild():
            raise RuntimeError("agent rebuild failed")

        manager._agent_factory = fail_agent_rebuild

        with pytest.raises(RuntimeError, match="agent rebuild failed"):
            await acp_agent.set_session_model(
                model_id="openai-codex:gpt-5.6-terra",
                session_id=state.session_id,
            )

        assert state.model == "gpt-5.5"
        assert state.agent is previous_agent

    @pytest.mark.asyncio
    async def test_successful_model_switch_disposes_previous_codex_runtime(self):
        class FakeCodexSession:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class FakeAgent:
            def __init__(self, model):
                self.model = model
                self.provider = "openai-codex"
                self.base_url = "https://chatgpt.com/backend-api/codex"
                self.api_mode = "codex_app_server"
                self._codex_session = FakeCodexSession()
                self.clients_released = False
                self.shared_resources_destroyed = False

            def release_clients(self):
                self.clients_released = True

            def close(self):
                self.shared_resources_destroyed = True

        previous_agent = FakeAgent("gpt-5.5")
        new_agent = FakeAgent("gpt-5.6-terra")
        agents = iter((previous_agent, new_agent))
        manager = SessionManager(agent_factory=lambda: next(agents), db=None)
        acp_agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd="/tmp")
        previous_codex_session = previous_agent._codex_session

        result = await acp_agent.set_session_model(
            model_id="openai-codex:gpt-5.6-terra",
            session_id=state.session_id,
        )

        assert isinstance(result, SetSessionModelResponse)
        assert state.agent is new_agent
        assert previous_codex_session.closed is True
        assert previous_agent._codex_session is None
        assert previous_agent.clients_released is True
        assert previous_agent.shared_resources_destroyed is False
        assert new_agent.clients_released is False

    @pytest.mark.asyncio
    async def test_persistence_failure_rolls_back_and_disposes_new_agent(self):
        class FakeAgent:
            def __init__(self, model):
                self.model = model
                self.provider = "openai-codex"
                self.base_url = "https://chatgpt.com/backend-api/codex"
                self.api_mode = "codex_app_server"
                self.clients_released = False
                self.shared_resources_destroyed = False

            def release_clients(self):
                self.clients_released = True

            def close(self):
                self.shared_resources_destroyed = True

        previous_agent = FakeAgent("gpt-5.5")
        new_agent = FakeAgent("gpt-5.6-terra")
        agents = iter((previous_agent, new_agent))
        manager = SessionManager(agent_factory=lambda: next(agents), db=None)
        acp_agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd="/tmp")
        manager.save_session = MagicMock(side_effect=(False, True))

        with pytest.raises(RuntimeError, match="Failed to persist ACP model switch"):
            await acp_agent.set_session_model(
                model_id="openai-codex:gpt-5.6-terra",
                session_id=state.session_id,
            )

        assert state.model == "gpt-5.5"
        assert state.agent is previous_agent
        assert previous_agent.clients_released is False
        assert previous_agent.shared_resources_destroyed is False
        assert new_agent.clients_released is True
        assert new_agent.shared_resources_destroyed is False
        assert manager.save_session.call_count == 2


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_keeps_cross_session_guard_after_binding(self, agent):
        """An unknown prompt must not weaken the foreign-session guard."""
        await agent.new_session(cwd=".")

        with pytest.raises(acp.RequestError, match="not owned by this process"):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="hello")],
                session_id="foreign-session",
            )

    @pytest.mark.asyncio
    async def test_prompt_runs_agent(self, agent):
        """The prompt method should call run_conversation on the agent."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Mock the agent's run_conversation
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Hello! How can I help?",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        })

        # Set up a mock connection
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()
        assert state.agent.tool_progress_callback is not None
        assert state.agent.step_callback is not None
        assert state.agent.stream_delta_callback is not None
        assert state.agent.reasoning_callback is not None
        assert state.agent.thinking_callback is None
    @pytest.mark.asyncio
    async def test_prompt_updates_history(self, agent):
        """After a prompt, session history should be updated."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        expected_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hey",
            "messages": expected_history,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert state.history == expected_history
    @pytest.mark.asyncio
    async def test_prompt_sends_final_message_update(self, agent):
        """The final response should be sent as an AgentMessageChunk."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "I can help with that!",
            "messages": [],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="help me")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        # session_update should include the final message (usage_update may follow it)
        mock_conn.session_update.assert_called()
        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        assert any(update.session_update == "agent_message_chunk" for update in updates)
    @pytest.mark.asyncio
    async def test_prompt_suppresses_cancel_interrupt_sentinel(self, agent):
        """ACP cancel status text should not be emitted as assistant output."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        sentinel = "Operation interrupted: waiting for model response (3.3s elapsed)."

        def mock_run(*args, **kwargs):
            state.cancel_event.set()
            return {
                "final_response": sentinel,
                "messages": list(state.history),
                "interrupted": True,
                "completed": False,
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("agent.title_generator.maybe_auto_title") as mock_title:
            prompt = [TextContentBlock(type="text", text="please do a long task")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_texts = [
            update.content.text
            for update in updates
            if update.session_update == "agent_message_chunk"
        ]
        assert resp.stop_reason == "cancelled"
        assert sentinel not in agent_texts
        assert not any(text.startswith("Operation interrupted:") for text in agent_texts)
        mock_title.assert_not_called()
    @pytest.mark.asyncio
    async def test_prompt_suppresses_final_response_when_cancel_wins(self, agent):
        """STOP is authoritative until the final ACP message is delivered."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        final_text = "The actual model answer arrived before cancellation settled."

        def mock_run(*args, **kwargs):
            state.cancel_event.set()
            return {
                "final_response": final_text,
                "messages": [],
                "interrupted": True,
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="finish if you can")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_texts = [
            update.content.text
            for update in updates
            if update.session_update == "agent_message_chunk"
        ]
        assert resp.stop_reason == "cancelled"
        assert final_text not in agent_texts
        assert agent_texts == []
    @pytest.mark.asyncio
    async def test_prompt_propagates_hermes_session_id_env(self, agent, monkeypatch):
        """ACP must propagate the originating session id to the agent loop
        via ``HERMES_SESSION_ID`` so tools that want to stamp side-effects
        with it (e.g. ``kanban_create``) can read the env var inside
        ``run_conversation``. The variable must be visible during the
        agent call AND restored afterwards so a re-used executor thread
        doesn't leak one session's id into another."""
        # Pre-condition: env is clean.
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        captured: dict[str, str | None] = {}

        def mock_run(user_message, conversation_history=None, task_id=None, **kwargs):
            # Inside the agent loop the env var must reflect the active
            # ACP session id. ``task_id`` is also the session id at this
            # boundary; assert both for symmetry.
            captured["env"] = os.environ.get("HERMES_SESSION_ID")
            captured["task_id"] = task_id
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert captured["env"] == new_resp.session_id, (
            "HERMES_SESSION_ID must be set to the originating ACP session id "
            "while the agent loop is running"
        )
        assert captured["task_id"] == new_resp.session_id
        # Post-condition: must be restored to the prior value (None here).
        assert os.environ.get("HERMES_SESSION_ID") is None, (
            "HERMES_SESSION_ID must be restored after the agent call so "
            "a re-used executor thread doesn't leak the id into the next "
            "session's tools"
        )
    @pytest.mark.asyncio
    async def test_prompt_restores_prior_hermes_session_id(self, agent, monkeypatch):
        """If the env already had HERMES_SESSION_ID set (e.g. nested
        agent loops), the prior value must be restored after the inner
        prompt completes — not popped, not left at the inner id."""
        monkeypatch.setenv("HERMES_SESSION_ID", "outer-sess")

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        captured: dict[str, str | None] = {}

        def mock_run(*args, **kwargs):
            captured["inner"] = os.environ.get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert captured["inner"] == new_resp.session_id
        # Outer scope must be restored.
        assert os.environ.get("HERMES_SESSION_ID") == "outer-sess"
    @pytest.mark.asyncio
    async def test_prompt_does_not_duplicate_streamed_final_message(self, agent):
        """If ACP already streamed response chunks, final_response should not be sent again."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            state.agent.stream_delta_callback("streamed answer")
            return {"final_response": "streamed answer", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_chunks = [update for update in updates if update.session_update == "agent_message_chunk"]
        assert len(agent_chunks) == 1
        assert agent_chunks[0].content.text == "streamed answer"
    @pytest.mark.asyncio
    async def test_prompt_delivers_transformed_response_after_streaming(self, agent):
        """If a transform_llm_output plugin hook modifies the response after
        streaming, ACP must deliver the transformed final_response so the
        appended/rewritten text reaches the client.
        """
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            state.agent.stream_delta_callback("original answer")
            return {
                "final_response": "original answer\n\n[plugin appended this]",
                "response_transformed": True,
                "messages": [],
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        # The streamed chunk and the post-stream transformed message should
        # both be present (final delivery is a separate update_agent_message_text
        # call carrying the full transformed text).
        all_texts = [
            getattr(getattr(u, "content", None), "text", None)
            for u in updates
        ]
        assert any(
            text and "[plugin appended this]" in text for text in all_texts
        ), f"expected transformed final to be delivered, got: {all_texts!r}"

    @pytest.mark.asyncio
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id

















# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0




    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
            force=True,
        )
        mock_save.assert_called_once_with(state.session_id)


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        before = str(resolve_agent_cwd())
        agent._handle_slash_command("/help", state)
        assert str(resolve_agent_cwd()) == before





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}


    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
