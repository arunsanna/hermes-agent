from types import SimpleNamespace
from unittest.mock import patch

import pytest
from acp.schema import McpServerStdio

from acp_adapter.orchestration import (
    apply_orchestration_tool_policy,
    enforce_session_mcp_registration,
    orchestration_meta,
    requested_disabled_toolsets,
    requested_orchestration_mode,
    without_reserved_switchboard_mcp,
)
from acp_adapter.server import HermesACPAgent


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _switchboard_tools() -> list[dict]:
    return [
        _tool("mcp__switchboard_orch__delegate"),
        _tool("mcp__switchboard_orch__wait_for"),
        _tool("mcp__switchboard_orch__agent_status"),
        _tool("mcp__switchboard_orch__cancel_agent"),
    ]


@pytest.fixture(autouse=True)
def _clean_contract_env(monkeypatch):
    monkeypatch.delenv("HERMES_ACP_ORCHESTRATION_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACP_DISABLED_TOOLSETS", raising=False)
    monkeypatch.delenv("HERMES_ACP_SWITCHBOARD_MCP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS", raising=False)


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
        "env": {},
        "timeout": 600.0,
    }


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
            return_value=_switchboard_tools(),
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
            "env": {},
            "timeout": 600.0,
        }
    }
    assert "mcp-switchboard_orch" in state.agent.enabled_toolsets
    assert state.agent._switchboard_orchestration_mcp_registration_verified is True


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
