from types import SimpleNamespace
from unittest.mock import patch

import pytest
from acp.schema import McpServerStdio

from acp_adapter.orchestration import (
    apply_switchboard_uat_direct_delegate_once,
    apply_orchestration_tool_policy,
    enforce_session_mcp_registration,
    orchestration_meta,
    restrict_verified_switchboard_request_tools,
    requested_disabled_toolsets,
    requested_orchestration_mode,
    switchboard_runtime_tool_block,
    without_switchboard_tool_search_bridge,
    without_reserved_switchboard_mcp,
)
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _switchboard_tools() -> list[dict]:
    return [
        _tool("mcp__switchboard_orch__delegate"),
        _tool("mcp__switchboard_orch__wait_for"),
        _tool("mcp__switchboard_orch__agent_status"),
        _tool("mcp__switchboard_orch__cancel_agent"),
    ]


def _responses_tool(name: str) -> dict:
    return {"type": "function", "name": name, "parameters": {"type": "object"}}


def _switchboard_responses_tools() -> list[dict]:
    return [
        _responses_tool("mcp__switchboard_orch__delegate"),
        _responses_tool("mcp__switchboard_orch__wait_for"),
        _responses_tool("mcp__switchboard_orch__agent_status"),
        _responses_tool("mcp__switchboard_orch__cancel_agent"),
    ]


def _tool_search_bridges() -> list[dict]:
    return [
        _tool("tool_search"),
        _tool("tool_describe"),
        _tool("tool_call"),
    ]


def _trusted_child_env() -> dict[str, str]:
    return {
        "SWITCHBOARD_GATEWAY_URL": "https://127.0.0.1:3030",
        "SWITCHBOARD_SESSION_ID": "sess-test-parent",
    }


@pytest.fixture(autouse=True)
def _clean_contract_env(monkeypatch):
    monkeypatch.delenv("HERMES_ACP_ORCHESTRATION_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACP_DISABLED_TOOLSETS", raising=False)
    monkeypatch.delenv("HERMES_ACP_SWITCHBOARD_MCP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv(
        "HERMES_ACP_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE", raising=False
    )
    for name, value in _trusted_child_env().items():
        monkeypatch.setenv(name, value)


def test_contract_is_inert_without_switchboard_request():
    kwargs = {"enabled_toolsets": ["hermes-acp"]}

    assert requested_orchestration_mode() is None
    assert apply_orchestration_tool_policy(kwargs) is None
    assert "disabled_toolsets" not in kwargs
    assert orchestration_meta(SimpleNamespace(tools=[])) is None


def test_invalid_requested_mode_fails_before_agent_construction(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "hybrid")

    with pytest.raises(RuntimeError, match="must be single, native, or switchboard"):
        apply_orchestration_tool_policy({})


def test_single_requires_hard_delegation_toolset_subtraction(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "single")

    with pytest.raises(RuntimeError, match="requires delegation"):
        apply_orchestration_tool_policy({})


def test_disabled_toolsets_are_deduplicated_in_stable_order(monkeypatch):
    monkeypatch.setenv(
        "HERMES_ACP_DISABLED_TOOLSETS", " delegation, terminal,delegation "
    )

    assert requested_disabled_toolsets() == ["delegation", "terminal"]


@pytest.mark.parametrize("mode", ["single", "switchboard"])
def test_non_native_mode_applies_delegation_subtraction(monkeypatch, mode):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", mode)
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    kwargs = {}

    assert apply_orchestration_tool_policy(kwargs) == mode
    assert kwargs["disabled_toolsets"] == ["delegation"]


def test_native_metadata_verifies_only_native_delegate_surface(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "native")
    agent = SimpleNamespace(
        disabled_toolsets=None,
        enabled_toolsets=["hermes-acp"],
        tools=[_tool("terminal"), _tool("delegate_task")],
    )

    meta = orchestration_meta(agent)

    assert meta == {
        "requestedMode": "native",
        "effectiveMode": "native",
        "disabledToolsets": [],
        "effectiveTools": ["delegate_task", "terminal"],
        "mcpServers": [],
        "mcpRegistrationVerified": False,
        "verified": True,
    }


def test_single_metadata_proves_delegate_and_switchboard_are_absent(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "single")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp"],
        tools=[_tool("terminal")],
    )

    meta = orchestration_meta(agent)

    assert meta["effectiveMode"] == "single"
    assert meta["verified"] is True
    assert "delegate_task" not in meta["effectiveTools"]


def test_switchboard_metadata_requires_mcp_and_no_native_delegate(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[_tool("terminal"), *_switchboard_tools()],
        _switchboard_orchestration_mcp_registration_verified=True,
    )

    meta = orchestration_meta(agent)

    assert meta["requestedMode"] == "switchboard"
    assert meta["effectiveMode"] == "switchboard"
    assert meta["mcpServers"] == ["switchboard_orch"]
    assert meta["mcpRegistrationVerified"] is True
    assert meta["verified"] is True


def test_switchboard_metadata_recovers_late_toolset_bookkeeping_loss(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp"],
        tools=_switchboard_tools(),
        _switchboard_orchestration_mcp_registration_verified=True,
    )

    meta = orchestration_meta(agent)

    assert meta["effectiveMode"] == "switchboard"
    assert meta["mcpServers"] == ["switchboard_orch"]
    assert meta["verified"] is True


def test_switchboard_model_schema_keeps_full_toolset(monkeypatch):
    """2026-08-16 owner decree: verified switchboard mode restricts nothing."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    schema = [
        *_switchboard_tools(),
        *_tool_search_bridges(),
        _tool("skill_view"),
        _tool("terminal"),
    ]

    visible = without_switchboard_tool_search_bridge(agent, schema)

    assert visible == schema


def test_switchboard_model_schema_survives_late_toolset_bookkeeping_loss(monkeypatch):
    """2026-08-16 owner decree: nothing is stripped even with bookkeeping loss."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp"],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    schema = [*_switchboard_tools(), *_tool_search_bridges(), _tool("terminal")]

    visible = without_switchboard_tool_search_bridge(agent, schema)

    assert visible == schema


def test_switchboard_uat_canary_never_forces_a_verified_request(monkeypatch):
    """2026-08-16 owner decree: forcing a tool choice is also forbidden."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE", "1")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp"],
        tools=_switchboard_tools(),
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    first_request = {"tools": _switchboard_tools(), "tool_choice": "auto"}

    assert apply_switchboard_uat_direct_delegate_once(agent, first_request) is False
    assert first_request == {"tools": _switchboard_tools(), "tool_choice": "auto"}

    following_parent_request = {"tools": _switchboard_tools(), "tool_choice": "auto"}
    assert (
        apply_switchboard_uat_direct_delegate_once(agent, following_parent_request)
        is False
    )
    assert following_parent_request == {
        "tools": _switchboard_tools(),
        "tool_choice": "auto",
    }


def test_switchboard_uat_canary_never_forces_responses_tool_shape(monkeypatch):
    """2026-08-16 owner decree: the Responses-shape request stays untouched too."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE", "1")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[*_switchboard_tools(), *_tool_search_bridges()],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    request = {
        "instructions": "Switchboard canary",
        "input": [{"role": "user", "content": "delegate"}],
        "tools": [*_switchboard_responses_tools(), _responses_tool("tool_call")],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    original_request = dict(request)

    assert apply_switchboard_uat_direct_delegate_once(agent, request) is False
    assert request == original_request


def test_verified_switchboard_request_surface_never_clamps(monkeypatch):
    """2026-08-16 owner decree: no fail-closed clamp, even on an incomplete surface."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        tools=_switchboard_tools()[:-1],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    request = {"tools": _switchboard_responses_tools(), "tool_choice": "auto"}
    original_request = dict(request)

    assert restrict_verified_switchboard_request_tools(agent, request) is None
    assert request == original_request


def test_switchboard_metadata_reports_a_broad_verified_surface_unclamped(monkeypatch):
    """2026-08-16 owner decree: metadata reports the full live surface, unclamped."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[*_switchboard_tools(), *_tool_search_bridges(), _tool("terminal")],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    original_tools = list(agent.tools)

    meta = orchestration_meta(agent)

    assert meta["effectiveMode"] == "switchboard"
    assert set(meta["effectiveTools"]) == {
        tool["function"]["name"] for tool in original_tools
    }
    # The agent's tool surface must not have been narrowed in place.
    assert agent.tools == original_tools
    assert not hasattr(agent, "valid_tool_names")


@pytest.mark.parametrize(
    "mode, verified, api_tools",
    [
        ("native", True, _switchboard_tools()),
        ("switchboard", False, _switchboard_tools()),
        ("switchboard", True, [_tool("mcp__switchboard_orch__delegate")]),
    ],
)
def test_switchboard_uat_canary_never_forces_or_clamps_any_schema(
    monkeypatch, mode, verified, api_tools
):
    """2026-08-16 owner decree: no case (including an incomplete verified
    surface) may force a tool choice or fail-closed clamp the request."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", mode)
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE", "1")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=_switchboard_tools(),
        _switchboard_orchestration_mcp_registration_verified=verified,
    )
    request = {"tools": api_tools, "tool_choice": "auto"}

    assert apply_switchboard_uat_direct_delegate_once(agent, request) is False
    assert request == {"tools": api_tools, "tool_choice": "auto"}


def test_verified_switchboard_runtime_gate_blocks_nothing(monkeypatch):
    """2026-08-16 owner decree: the runtime gate never denies any tool."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp"],
        tools=[_tool("read_file"), *_switchboard_tools()],
        _switchboard_orchestration_mcp_registration_verified=True,
    )

    assert switchboard_runtime_tool_block(agent, "read_file") is None
    assert (
        switchboard_runtime_tool_block(
            agent, "mcp__switchboard_orch__delegate"
        )
        is None
    )


def test_unverified_switchboard_runtime_gate_stays_inert(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[_tool("read_file"), *_switchboard_tools()],
        _switchboard_orchestration_mcp_registration_verified=False,
    )

    assert switchboard_runtime_tool_block(agent, "read_file") is None


def test_direct_registry_dispatch_never_blocked_by_switchboard_runtime_gate(
    monkeypatch,
):
    """2026-08-16 owner decree: direct dispatch is never denied in any mode."""
    import model_tools

    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[_tool("read_file"), *_switchboard_tools()],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    controller = "mcp__switchboard_orch__delegate"

    with (
        patch("model_tools.registry.dispatch", return_value='{"ok": true}') as dispatch,
        patch("model_tools._emit_post_tool_call_hook"),
    ):
        first = model_tools.handle_function_call(
            "read_file",
            {"path": "/tmp/nope"},
            parent_agent=agent,
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )
        second = model_tools.handle_function_call(
            controller,
            {},
            parent_agent=agent,
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert first == '{"ok": true}'
    assert second == '{"ok": true}'
    assert dispatch.call_count == 2
    assert dispatch.call_args_list[0].args[0] == "read_file"
    assert dispatch.call_args_list[1].args[0] == controller


def test_direct_dispatch_gate_failure_does_not_narrow_native_mode(monkeypatch):
    """A broken ACP runtime guard must not affect non-managed callers."""
    import model_tools

    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "native")
    agent = SimpleNamespace()

    with (
        patch(
            "acp_adapter.orchestration.switchboard_runtime_tool_block",
            side_effect=RuntimeError("simulated ACP import failure"),
        ),
        patch("model_tools.registry.dispatch", return_value='{"ok": true}') as dispatch,
        patch("model_tools._emit_post_tool_call_hook"),
    ):
        result = model_tools.handle_function_call(
            "read_file",
            {"path": "/tmp/normal"},
            parent_agent=agent,
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert result == '{"ok": true}'
    dispatch.assert_called_once()


def test_unverified_switchboard_schema_retains_generic_tool_search(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        _switchboard_orchestration_mcp_registration_verified=False,
    )
    schema = [*_switchboard_tools(), *_tool_search_bridges()]

    assert without_switchboard_tool_search_bridge(agent, schema) == schema


@pytest.mark.parametrize("mode", ["single", "native"])
def test_non_switchboard_schema_retains_generic_tool_search(monkeypatch, mode):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", mode)
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        _switchboard_orchestration_mcp_registration_verified=True,
    )
    schema = [*_switchboard_tools(), *_tool_search_bridges()]

    assert without_switchboard_tool_search_bridge(agent, schema) == schema


@pytest.mark.parametrize(
    "switchboard_tools",
    [
        [_tool("mcp__switchboard_orch__delegate")],
        [*_switchboard_tools(), _tool("mcp__switchboard_orch__unexpected")],
    ],
)
def test_switchboard_metadata_rejects_partial_or_extra_reserved_tools(
    monkeypatch, switchboard_tools
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=switchboard_tools,
        _switchboard_orchestration_mcp_registration_verified=True,
    )

    meta = orchestration_meta(agent)

    assert meta["verified"] is False
    assert "switchboard_orch MCP tools" in meta["mismatchReason"]


def test_switchboard_metadata_rejects_untrusted_live_registration(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=_switchboard_tools(),
    )

    meta = orchestration_meta(agent)

    assert meta["verified"] is False
    assert "live registration was not trusted" in meta["mismatchReason"]


def test_managed_mode_removes_reserved_server_from_global_config(monkeypatch):
    configured = {
        "switchboard_orch": {"command": "/tmp/untrusted"},
        "other": {"url": "https://example.invalid/mcp"},
    }

    assert without_reserved_switchboard_mcp(configured) == configured

    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    assert without_reserved_switchboard_mcp(configured) == {
        "other": {"url": "https://example.invalid/mcp"}
    }


def test_switchboard_metadata_reports_mismatch_when_mcp_is_missing(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp"],
        tools=[_tool("terminal")],
    )

    meta = orchestration_meta(agent)

    assert meta["effectiveMode"] == "single"
    assert meta["verified"] is False
    assert "switchboard_orch MCP server is absent" in meta["mismatchReason"]


def test_switchboard_metadata_rejects_registered_server_without_effective_tools(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    agent = SimpleNamespace(
        disabled_toolsets=["delegation"],
        enabled_toolsets=["hermes-acp", "mcp-switchboard_orch"],
        tools=[_tool("terminal")],
    )

    meta = orchestration_meta(agent)

    assert meta["effectiveMode"] == "single"
    assert meta["verified"] is False
    assert "switchboard_orch MCP tools are incomplete" in meta["mismatchReason"]


def test_switchboard_reserved_mcp_uses_trusted_launcher_and_600_second_timeout(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")

    config = enforce_session_mcp_registration(
        "switchboard_orch",
        {
            "command": "/private/tmp/orch-launcher",
            "args": [],
            "env": {},
        },
        is_stdio=True,
    )

    assert config == {
        "command": "/private/tmp/orch-launcher",
        "args": [],
        "env": _trusted_child_env(),
        "timeout": 600.0,
    }


@pytest.mark.parametrize(
    "missing_env",
    ["SWITCHBOARD_GATEWAY_URL", "SWITCHBOARD_SESSION_ID"],
)
def test_switchboard_reserved_mcp_requires_trusted_parent_environment(
    monkeypatch, missing_env
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")
    monkeypatch.delenv(missing_env)

    with pytest.raises(RuntimeError, match=missing_env):
        enforce_session_mcp_registration(
            "switchboard_orch",
            {
                "command": "/private/tmp/orch-launcher",
                "args": [],
                "env": {},
            },
            is_stdio=True,
        )


@pytest.mark.asyncio
async def test_acp_registration_passes_hardened_switchboard_config(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")
    state = SimpleNamespace(
        session_id="acp-1",
        agent=SimpleNamespace(
            enabled_toolsets=["hermes-acp"],
            disabled_toolsets=["delegation"],
            tools=[],
            valid_tool_names=set(),
        ),
    )
    server = object.__new__(HermesACPAgent)
    captured = {}

    def _capture(config_map):
        captured.update(config_map)
        return ["mcp__switchboard_orch__delegate"]

    with (
        patch("tools.mcp_tool.register_mcp_servers", side_effect=_capture),
        patch(
            "tools.mcp_tool.registered_mcp_server_matches_config",
            return_value=True,
        ),
        patch(
            "model_tools.get_tool_definitions",
            return_value=[
                *_switchboard_tools(),
                *_tool_search_bridges(),
                _tool("skill_view"),
                _tool("terminal"),
            ],
        ),
        patch("agent.memory_manager.inject_memory_provider_tools"),
    ):
        await server._register_session_mcp_servers(
            state,
            [
                McpServerStdio(
                    name="switchboard_orch",
                    command="/private/tmp/orch-launcher",
                    args=[],
                    env=[],
                )
            ],
        )

    assert captured == {
        "switchboard_orch": {
            "command": "/private/tmp/orch-launcher",
            "args": [],
            "env": _trusted_child_env(),
            "timeout": 600.0,
        }
    }
    assert "mcp-switchboard_orch" in state.agent.enabled_toolsets
    assert state.agent._switchboard_orchestration_mcp_registration_verified is True
    # 2026-08-16 owner decree: registration must not narrow the model-facing
    # tool surface -- the full mocked catalog (controllers, bridges, skills,
    # terminal) survives unchanged.
    assert {tool["function"]["name"] for tool in state.agent.tools} == {
        *{tool["function"]["name"] for tool in _switchboard_tools()},
        *{tool["function"]["name"] for tool in _tool_search_bridges()},
        "skill_view",
        "terminal",
    }


@pytest.mark.asyncio
async def test_acp_registration_keeps_switchboard_tools_in_model_schema(monkeypatch):
    """The registry here only ever contains the 4 switchboard tools, so this
    passes regardless of restriction logic; kept as coverage of the live ACP
    registration path itself."""
    import model_tools
    from acp_adapter.orchestration import _SWITCHBOARD_MCP_TOOL_NAMES
    from tools.registry import ToolRegistry

    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")

    registry = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", registry)
    monkeypatch.setattr(model_tools, "registry", registry)
    model_tools._clear_tool_defs_cache()

    state = SimpleNamespace(
        session_id="acp-model-schema",
        agent=SimpleNamespace(
            enabled_toolsets=["hermes-acp"],
            disabled_toolsets=["delegation"],
            tools=[],
            valid_tool_names=set(),
        ),
    )
    server = object.__new__(HermesACPAgent)

    def _register(config_map):
        assert set(config_map) == {"switchboard_orch"}
        for name in _SWITCHBOARD_MCP_TOOL_NAMES:
            registry.register(
                name=name,
                toolset="mcp-switchboard_orch",
                schema=_tool(name),
                handler=lambda *_args, **_kwargs: "{}",
            )
        return sorted(_SWITCHBOARD_MCP_TOOL_NAMES)

    try:
        with (
            patch("tools.mcp_tool.register_mcp_servers", side_effect=_register),
            patch(
                "tools.mcp_tool.registered_mcp_server_matches_config",
                return_value=True,
            ),
            patch("agent.memory_manager.inject_memory_provider_tools"),
        ):
            await server._register_session_mcp_servers(
                state,
                [
                    McpServerStdio(
                        name="switchboard_orch",
                        command="/private/tmp/orch-launcher",
                        args=[],
                        env=[],
                    )
                ],
            )

        model_tool_names = {
            tool["function"]["name"] for tool in state.agent.tools
        }
        assert model_tool_names == _SWITCHBOARD_MCP_TOOL_NAMES
        assert state.agent.valid_tool_names == _SWITCHBOARD_MCP_TOOL_NAMES
        assert orchestration_meta(state.agent)["verified"] is True
    finally:
        model_tools._clear_tool_defs_cache()


@pytest.mark.asyncio
async def test_session_model_switch_restores_verified_switchboard_mcp_surface(
    tmp_path, monkeypatch
):
    """A replacement agent must receive the exact safe ACP MCP contract."""
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")

    def _agent_factory():
        return SimpleNamespace(
            model="initial-model",
            provider="openrouter",
            base_url="https://openrouter.example/v1",
            api_mode="chat_completions",
            enabled_toolsets=["hermes-acp"],
            disabled_toolsets=["delegation"],
            tools=[],
            valid_tool_names=set(),
        )

    manager = SessionManager(
        agent_factory=_agent_factory,
        db=SessionDB(tmp_path / "state.db"),
    )
    server = HermesACPAgent(session_manager=manager)
    server._schedule_mcp_late_refresh = lambda _state: None
    registered = []

    def _register(config_map):
        registered.append(config_map)
        return sorted(
            tool["function"]["name"] for tool in _switchboard_tools()
        )

    with (
        patch("tools.mcp_tool.register_mcp_servers", side_effect=_register),
        patch(
            "tools.mcp_tool.registered_mcp_server_matches_config",
            return_value=True,
        ),
        patch(
            "model_tools.get_tool_definitions",
            return_value=[*_switchboard_tools(), *_tool_search_bridges(), _tool("terminal")],
        ),
        patch("agent.memory_manager.inject_memory_provider_tools"),
    ):
        response = await server.new_session(
            cwd="/tmp/project",
            mcp_servers=[
                McpServerStdio(
                    name="switchboard_orch",
                    command="/private/tmp/orch-launcher",
                    args=[],
                    env=[],
                )
            ],
        )
        state = manager.get_session(response.session_id)
        result = await server.set_session_model(
            model_id="openrouter:replacement-model",
            session_id=response.session_id,
        )

    assert result is not None
    expected_config = {
        "switchboard_orch": {
            "command": "/private/tmp/orch-launcher",
            "args": [],
            "env": _trusted_child_env(),
            "timeout": 600.0,
        }
    }
    assert registered == [expected_config, expected_config]
    assert state._acp_session_mcp_server_configs == expected_config
    assert state.agent._switchboard_orchestration_mcp_registration_verified is True
    # 2026-08-16 owner decree: the replacement agent's tool surface must not
    # be narrowed by the model switch -- the full mocked catalog survives.
    assert {tool["function"]["name"] for tool in state.agent.tools} == {
        *{tool["function"]["name"] for tool in _switchboard_tools()},
        *{tool["function"]["name"] for tool in _tool_search_bridges()},
        "terminal",
    }
    assert orchestration_meta(state.agent)["effectiveMode"] == "switchboard"
    assert result.field_meta["switchboardOrchestration"] == orchestration_meta(
        state.agent
    )
    assert result.model_dump(by_alias=True)["_meta"]["switchboardOrchestration"] == (
        orchestration_meta(state.agent)
    )


@pytest.mark.asyncio
async def test_session_model_switch_refuses_incomplete_switchboard_surface(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")

    def _agent_factory():
        return SimpleNamespace(
            model="initial-model",
            provider="openrouter",
            base_url="https://openrouter.example/v1",
            api_mode="chat_completions",
            enabled_toolsets=["hermes-acp"],
            disabled_toolsets=["delegation"],
            tools=[],
            valid_tool_names=set(),
        )

    manager = SessionManager(
        agent_factory=_agent_factory,
        db=SessionDB(tmp_path / "state.db"),
    )
    server = HermesACPAgent(session_manager=manager)
    server._schedule_mcp_late_refresh = lambda _state: None

    with (
        patch("tools.mcp_tool.register_mcp_servers", return_value=[]),
        patch(
            "tools.mcp_tool.registered_mcp_server_matches_config",
            return_value=True,
        ),
        patch(
            "model_tools.get_tool_definitions",
            side_effect=[_switchboard_tools(), _switchboard_tools()[:1]],
        ),
        patch("agent.memory_manager.inject_memory_provider_tools"),
    ):
        response = await server.new_session(
            cwd="/tmp/project",
            mcp_servers=[
                McpServerStdio(
                    name="switchboard_orch",
                    command="/private/tmp/orch-launcher",
                    args=[],
                    env=[],
                )
            ],
        )
        state = manager.get_session(response.session_id)
        original_agent = state.agent
        with pytest.raises(RuntimeError, match="restore ACP MCP servers"):
            await server.set_session_model(
                model_id="openrouter:replacement-model",
                session_id=response.session_id,
            )

    assert state.agent is original_agent
    assert state.model == "initial-model"


@pytest.mark.asyncio
async def test_acp_registration_rejects_preexisting_name_collision(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    monkeypatch.setenv(
        "HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/private/tmp/orch-launcher"
    )
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")
    state = SimpleNamespace(
        session_id="acp-collision",
        agent=SimpleNamespace(
            enabled_toolsets=["hermes-acp"],
            disabled_toolsets=["delegation"],
            tools=_switchboard_tools(),
            valid_tool_names=set(),
        ),
    )
    server = object.__new__(HermesACPAgent)

    with (
        patch("tools.mcp_tool.register_mcp_servers", return_value=[]),
        patch(
            "tools.mcp_tool.registered_mcp_server_matches_config",
            return_value=False,
        ),
    ):
        await server._register_session_mcp_servers(
            state,
            [
                McpServerStdio(
                    name="switchboard_orch",
                    command="/private/tmp/orch-launcher",
                    args=[],
                    env=[],
                )
            ],
        )

    assert state.agent._switchboard_orchestration_mcp_registration_verified is False
    meta = orchestration_meta(state.agent)
    assert meta["verified"] is False
    assert "live registration was not trusted" in meta["mismatchReason"]


@pytest.mark.asyncio
async def test_managed_fork_fails_closed_before_reusing_parent_mcp(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    server = object.__new__(HermesACPAgent)
    server.session_manager = SimpleNamespace(
        fork_session=lambda *_args, **_kwargs: pytest.fail(
            "managed fork must not reach SessionManager"
        )
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        await server.fork_session(cwd="/tmp", session_id="parent", mcp_servers=[])


@pytest.mark.parametrize(
    ("config", "is_stdio", "error"),
    [
        (
            {"command": "/tmp/untrusted", "args": [], "env": {}},
            True,
            "trusted launcher",
        ),
        (
            {"command": "/tmp/trusted", "args": ["--credential"], "env": {}},
            True,
            "does not accept arguments",
        ),
        (
            {"command": "/tmp/trusted", "args": [], "env": {"TOKEN": "secret"}},
            True,
            "caller-provided environment",
        ),
        (
            {"url": "https://example.invalid/mcp", "headers": {}},
            False,
            "stdio transport",
        ),
    ],
)
def test_switchboard_reserved_mcp_rejects_untrusted_registration(
    monkeypatch,
    config,
    is_stdio,
    error,
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/tmp/trusted")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", "600")

    with pytest.raises(RuntimeError, match=error):
        enforce_session_mcp_registration("switchboard_orch", config, is_stdio=is_stdio)


@pytest.mark.parametrize("timeout", ["", "599", "nan", "infinity", "invalid"])
def test_switchboard_reserved_mcp_rejects_wrong_timeout(monkeypatch, timeout):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "switchboard")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_COMMAND", "/tmp/trusted")
    monkeypatch.setenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", timeout)

    with pytest.raises(RuntimeError, match="must be 600"):
        enforce_session_mcp_registration(
            "switchboard_orch",
            {"command": "/tmp/trusted", "args": [], "env": {}},
            is_stdio=True,
        )


@pytest.mark.parametrize("mode", ["single", "native"])
def test_switchboard_reserved_mcp_is_rejected_outside_switchboard_mode(
    monkeypatch, mode
):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", mode)

    with pytest.raises(RuntimeError, match="reserved for switchboard mode"):
        enforce_session_mcp_registration(
            "switchboard_orch",
            {"command": "/tmp/trusted", "args": [], "env": {}},
            is_stdio=True,
        )


def test_session_meta_merges_provenance_and_effective_policy(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "single")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    state = SimpleNamespace(
        session_id="acp-1",
        agent=SimpleNamespace(
            session_id="hermes-1",
            disabled_toolsets=["delegation"],
            enabled_toolsets=["hermes-acp"],
            tools=[_tool("terminal")],
        ),
    )
    server = object.__new__(HermesACPAgent)
    server._provenance_meta = lambda *args, **kwargs: {
        "hermes": {"sessionProvenance": {"acpSessionId": "acp-1"}}
    }

    meta = server._session_meta(state)

    assert meta["hermes"]["sessionProvenance"]["acpSessionId"] == "acp-1"
    assert meta["switchboardOrchestration"]["effectiveMode"] == "single"
    assert meta["switchboardOrchestration"]["verified"] is True


@pytest.mark.asyncio
async def test_session_info_update_keeps_effective_orchestration_evidence(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_ORCHESTRATION_MODE", "single")
    monkeypatch.setenv("HERMES_ACP_DISABLED_TOOLSETS", "delegation")
    state = SimpleNamespace(
        session_id="acp-1",
        agent=SimpleNamespace(
            session_id="hermes-1",
            disabled_toolsets=["delegation"],
            enabled_toolsets=["hermes-acp"],
            tools=[_tool("terminal")],
        ),
    )
    updates = []

    async def _session_update(**kwargs):
        updates.append(kwargs)

    server = object.__new__(HermesACPAgent)
    server._conn = SimpleNamespace(session_update=_session_update)
    server._provenance_meta = lambda *args, **kwargs: {
        "hermes": {"sessionProvenance": {"acpSessionId": "acp-1"}}
    }
    server.session_manager = SimpleNamespace(
        _get_db=lambda: SimpleNamespace(
            get_session=lambda _session_id: {"title": "Task"}
        ),
        peek_session=lambda _session_id: state,
    )

    await server._send_session_info_update("acp-1")

    assert len(updates) == 1
    update = updates[0]["update"]
    assert update.field_meta["switchboardOrchestration"] == {
        "requestedMode": "single",
        "effectiveMode": "single",
        "disabledToolsets": ["delegation"],
        "effectiveTools": ["terminal"],
        "mcpServers": [],
        "mcpRegistrationVerified": False,
        "verified": True,
    }
