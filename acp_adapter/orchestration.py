"""Session-scoped Switchboard orchestration enforcement for Hermes ACP.

The environment variables consumed here are an internal bridge contract, not
user-facing Hermes configuration.  Switchboard launches one ``hermes-acp``
process per SDK session and pins the requested orchestration owner before the
agent is constructed.  Hermes then reports the *observed* tool surface in ACP
``_meta`` so the client can fail closed instead of trusting launch intent.
"""

from __future__ import annotations

import math
import os
from typing import Any


_MODE_ENV = "HERMES_ACP_ORCHESTRATION_MODE"
_DISABLED_TOOLSETS_ENV = "HERMES_ACP_DISABLED_TOOLSETS"
_SWITCHBOARD_MCP_COMMAND_ENV = "HERMES_ACP_SWITCHBOARD_MCP_COMMAND"
_SWITCHBOARD_MCP_TOOL_TIMEOUT_ENV = "HERMES_ACP_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS"
_VALID_MODES = frozenset({"single", "native", "switchboard"})
_SWITCHBOARD_MCP_SERVER = "switchboard_orch"
_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS = 600.0
_SWITCHBOARD_MCP_TOOL_NAMES = frozenset({
    "mcp__switchboard_orch__delegate",
    "mcp__switchboard_orch__wait_for",
    "mcp__switchboard_orch__agent_status",
    "mcp__switchboard_orch__cancel_agent",
})


def requested_orchestration_mode() -> str | None:
    """Return Switchboard's requested owner, rejecting an invalid contract."""
    raw = (os.environ.get(_MODE_ENV) or "").strip().lower()
    if not raw:
        return None
    if raw not in _VALID_MODES:
        raise RuntimeError(
            f"{_MODE_ENV} must be single, native, or switchboard (got {raw!r})"
        )
    return raw


def requested_disabled_toolsets() -> list[str]:
    """Parse the bridge-only disabled-toolset allowlist deterministically."""
    result: list[str] = []
    for item in (os.environ.get(_DISABLED_TOOLSETS_ENV) or "").split(","):
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result


def without_reserved_switchboard_mcp(
    servers: dict[str, Any] | None,
) -> dict[str, Any]:
    """Remove the bridge-reserved MCP name from process-global config.

    Hermes MCP discovery is process-global and keyed only by server name. A
    configured or portable server named ``switchboard_orch`` could otherwise
    connect before ACP registers Switchboard's session-bound launcher, after
    which name-idempotent registration would silently retain the wrong
    command and credential scope. In a managed bridge process the reserved
    server may therefore enter only through ACP ``session/new``.
    """
    result = dict(servers or {})
    if requested_orchestration_mode() is not None:
        result.pop(_SWITCHBOARD_MCP_SERVER, None)
    return result


def apply_orchestration_tool_policy(agent_kwargs: dict[str, Any]) -> str | None:
    """Apply the requested hard toolset subtraction before agent creation."""
    mode = requested_orchestration_mode()
    if mode is None:
        return None

    disabled = requested_disabled_toolsets()
    if mode in {"single", "switchboard"} and "delegation" not in disabled:
        raise RuntimeError(
            f"{mode} mode requires delegation in {_DISABLED_TOOLSETS_ENV}"
        )
    agent_kwargs["disabled_toolsets"] = disabled or None
    return mode


def enforce_session_mcp_registration(
    name: str,
    config: dict[str, Any],
    *,
    is_stdio: bool,
) -> dict[str, Any]:
    """Protect the reserved Switchboard MCP registration contract.

    These environment variables are a private parent/child bridge, not
    user-facing Hermes settings.  Switchboard passes a random launcher path;
    the launcher alone receives the session credential.  The ACP parent must
    neither accept a caller-supplied replacement command nor forward arbitrary
    arguments/environment into that reserved child.
    """
    mode = requested_orchestration_mode()
    if mode is None or name != _SWITCHBOARD_MCP_SERVER:
        return config
    if mode != "switchboard":
        raise RuntimeError(
            f"{_SWITCHBOARD_MCP_SERVER} is reserved for switchboard mode"
        )
    if not is_stdio:
        raise RuntimeError(f"{_SWITCHBOARD_MCP_SERVER} must use the stdio transport")

    expected_command = (os.environ.get(_SWITCHBOARD_MCP_COMMAND_ENV) or "").strip()
    if not expected_command:
        raise RuntimeError(f"{_SWITCHBOARD_MCP_COMMAND_ENV} is required")
    if config.get("command") != expected_command:
        raise RuntimeError(
            f"{_SWITCHBOARD_MCP_SERVER} command does not match the trusted launcher"
        )
    if config.get("args"):
        raise RuntimeError(f"{_SWITCHBOARD_MCP_SERVER} does not accept arguments")
    if config.get("env"):
        raise RuntimeError(
            f"{_SWITCHBOARD_MCP_SERVER} does not accept caller-provided environment"
        )

    raw_timeout = (os.environ.get(_SWITCHBOARD_MCP_TOOL_TIMEOUT_ENV) or "").strip()
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{_SWITCHBOARD_MCP_TOOL_TIMEOUT_ENV} must be "
            f"{_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS:g}"
        ) from exc
    if not math.isfinite(timeout) or timeout != _SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"{_SWITCHBOARD_MCP_TOOL_TIMEOUT_ENV} must be "
            f"{_SWITCHBOARD_MCP_TOOL_TIMEOUT_SECONDS:g}"
        )

    hardened = dict(config)
    hardened["command"] = expected_command
    hardened["args"] = []
    hardened["env"] = {}
    hardened["timeout"] = timeout
    return hardened


def _tool_names(agent: Any) -> list[str]:
    names: set[str] = set()
    for tool in getattr(agent, "tools", None) or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return sorted(names)


def _mcp_server_names(agent: Any) -> list[str]:
    prefix = "mcp-"
    return sorted({
        value[len(prefix) :]
        for value in (getattr(agent, "enabled_toolsets", None) or [])
        if isinstance(value, str)
        and value.startswith(prefix)
        and len(value) > len(prefix)
    })


def orchestration_meta(agent: Any) -> dict[str, Any] | None:
    """Build the ACP extension proving the session's effective tool owner."""
    requested = requested_orchestration_mode()
    if requested is None:
        return None

    disabled = sorted(set(getattr(agent, "disabled_toolsets", None) or []))
    tools = _tool_names(agent)
    mcp_servers = _mcp_server_names(agent)
    has_native_delegate = "delegate_task" in tools
    switchboard_tool_prefix = f"mcp__{_SWITCHBOARD_MCP_SERVER}__"
    switchboard_tools = {
        name for name in tools if name.startswith(switchboard_tool_prefix)
    }
    has_switchboard_server = _SWITCHBOARD_MCP_SERVER in mcp_servers
    exact_switchboard_tools = switchboard_tools == _SWITCHBOARD_MCP_TOOL_NAMES
    trusted_switchboard_registration = bool(
        getattr(
            agent,
            "_switchboard_orchestration_mcp_registration_verified",
            False,
        )
    )
    has_switchboard = (
        has_switchboard_server
        and exact_switchboard_tools
        and trusted_switchboard_registration
    )

    if has_native_delegate:
        effective = "native"
    elif has_switchboard:
        effective = "switchboard"
    else:
        effective = "single"

    reasons: list[str] = []
    if requested in {"single", "switchboard"} and "delegation" not in disabled:
        reasons.append("delegation toolset was not disabled")
    if requested in {"single", "switchboard"} and has_native_delegate:
        reasons.append("delegate_task remains effective")
    if requested == "switchboard" and not has_switchboard_server:
        reasons.append("switchboard_orch MCP server is absent")
    if requested == "switchboard" and not exact_switchboard_tools:
        missing = sorted(_SWITCHBOARD_MCP_TOOL_NAMES - switchboard_tools)
        unexpected = sorted(switchboard_tools - _SWITCHBOARD_MCP_TOOL_NAMES)
        if missing:
            reasons.append(
                "switchboard_orch MCP tools are incomplete: missing "
                + ", ".join(missing)
            )
        if unexpected:
            reasons.append(
                "switchboard_orch MCP tools contain unexpected names: "
                + ", ".join(unexpected)
            )
    if requested == "switchboard" and not trusted_switchboard_registration:
        reasons.append("switchboard_orch live registration was not trusted")
    if requested != "switchboard" and has_switchboard:
        reasons.append("switchboard_orch MCP server is unexpectedly present")
    if requested == "native" and not has_native_delegate:
        reasons.append("delegate_task is absent")
    if effective != requested:
        reasons.append(f"effective mode is {effective}, requested {requested}")

    result: dict[str, Any] = {
        "requestedMode": requested,
        "effectiveMode": effective,
        "disabledToolsets": disabled,
        "effectiveTools": tools,
        "mcpServers": mcp_servers,
        "mcpRegistrationVerified": trusted_switchboard_registration,
        "verified": not reasons,
    }
    if reasons:
        result["mismatchReason"] = "; ".join(dict.fromkeys(reasons))
    return result
