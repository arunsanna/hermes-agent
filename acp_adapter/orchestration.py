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
# A private, disposable canary switch used only by Switchboard's ACP launch
# contract.  It intentionally accepts exactly ``1`` and is not Hermes config.
_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE_ENV = (
    "HERMES_ACP_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE"
)
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


def without_switchboard_tool_search_bridge(
    agent: Any,
    tool_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restrict a verified managed parent to Switchboard controller tools.

    The Switchboard controller controls are a four-tool, same-turn protocol,
    not optional catalog entries.  Leaving ``tool_search``/``tool_describe``/
    ``tool_call`` beside them invites smaller models to proxy a control through
    the generic bridge, which cannot safely dispatch a non-deferrable tool.
    Other local tools (including skills) similarly give a managed parent an
    alternate execution path instead of forcing Switchboard delegation.

    This only alters a model-facing schema after all three boundaries hold:
    Switchboard mode was requested, the session-bound registration was
    verified, and the effective reserved namespace is exactly the trusted
    four-tool surface.  Native/single sessions and lookalike plugin tools keep
    the normal progressive-disclosure behavior.
    """
    if requested_orchestration_mode() != "switchboard":
        return tool_defs
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return tool_defs

    enabled_toolsets = set(getattr(agent, "enabled_toolsets", None) or [])
    if f"mcp-{_SWITCHBOARD_MCP_SERVER}" not in enabled_toolsets:
        return tool_defs

    names = {
        function.get("name")
        for tool in tool_defs
        if isinstance(tool, dict)
        for function in [tool.get("function")]
        if isinstance(function, dict) and isinstance(function.get("name"), str)
    }
    switchboard_names = {
        name
        for name in names
        if name.startswith(f"mcp__{_SWITCHBOARD_MCP_SERVER}__")
    }
    if switchboard_names != _SWITCHBOARD_MCP_TOOL_NAMES:
        return tool_defs

    return [
        tool
        for tool in tool_defs
        if (tool.get("function") or {}).get("name") in _SWITCHBOARD_MCP_TOOL_NAMES
    ]


def apply_switchboard_uat_direct_delegate_once(
    agent: Any,
    api_kwargs: dict[str, Any],
) -> bool:
    """Force one verified managed-parent request to the delegate controller.

    This is deliberately a private ACP canary, rather than a configurable
    model-routing feature: it is active only when Switchboard explicitly
    launches the parent with the exact value ``1``.  The per-agent marker is
    consumed before dispatch so a retry rebuilt by the conversation loop, and
    every later parent response, revert to normal managed-controller choice.

    Both the live agent surface and the outgoing OpenAI tool list must be the
    exact trusted four-tool Switchboard registration.  A native, single,
    unverified, partial, or lookalike registration therefore cannot force an
    arbitrary provider tool choice.
    """
    if os.environ.get(_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE_ENV) != "1":
        return False
    if getattr(agent, "_switchboard_uat_direct_delegate_once_consumed", False):
        return False
    if requested_orchestration_mode() != "switchboard":
        return False
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return False

    enabled_toolsets = set(getattr(agent, "enabled_toolsets", None) or [])
    if f"mcp-{_SWITCHBOARD_MCP_SERVER}" not in enabled_toolsets:
        return False
    if set(_tool_names(agent)) != _SWITCHBOARD_MCP_TOOL_NAMES:
        return False

    api_tools = api_kwargs.get("tools")
    if not isinstance(api_tools, list):
        return False
    api_tool_names = {
        function.get("name")
        for tool in api_tools
        if isinstance(tool, dict)
        for function in [tool.get("function")]
        if isinstance(function, dict) and isinstance(function.get("name"), str)
    }
    if api_tool_names != _SWITCHBOARD_MCP_TOOL_NAMES:
        return False

    # Consume before the provider call: retry rebuilding must never make a
    # second logical request forcibly delegate, while a streaming fallback
    # retains these same kwargs for its non-streaming transport retry.
    setattr(agent, "_switchboard_uat_direct_delegate_once_consumed", True)
    api_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": "mcp__switchboard_orch__delegate"},
    }
    api_kwargs["parallel_tool_calls"] = False
    return True


def switchboard_runtime_tool_block(agent: Any, tool_name: str) -> str | None:
    """Return a denial for a non-controller call in a verified managed parent.

    The model-facing schema is helpful guidance, but it is not an execution
    boundary: a provider can retain an earlier schema, emit a hallucinated
    function, or reach a direct registry dispatch path.  Once the session has
    proved the exact trusted Switchboard MCP surface, the parent may execute
    only those four controller calls.  The dispatchers consume this helper
    before local special cases and registry dispatch.

    It deliberately stays inert for native, single, and unverified sessions.
    Those modes retain Hermes' normal tool policy and are not accidentally
    narrowed by a bridge-only environment variable.
    """
    if requested_orchestration_mode() != "switchboard":
        return None
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return None

    enabled_toolsets = set(getattr(agent, "enabled_toolsets", None) or [])
    if f"mcp-{_SWITCHBOARD_MCP_SERVER}" not in enabled_toolsets:
        return None

    switchboard_prefix = f"mcp__{_SWITCHBOARD_MCP_SERVER}__"
    effective_switchboard_tools = {
        name for name in _tool_names(agent) if name.startswith(switchboard_prefix)
    }
    if effective_switchboard_tools != _SWITCHBOARD_MCP_TOOL_NAMES:
        return None

    if tool_name in _SWITCHBOARD_MCP_TOOL_NAMES:
        return None
    return (
        "Switchboard orchestration runtime policy permits only the verified "
        "controller tools: "
        + ", ".join(sorted(_SWITCHBOARD_MCP_TOOL_NAMES))
        + ". The requested tool was not executed."
    )


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
