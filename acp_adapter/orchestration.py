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
_SWITCHBOARD_GATEWAY_URL_ENV = "SWITCHBOARD_GATEWAY_URL"
_SWITCHBOARD_SESSION_ID_ENV = "SWITCHBOARD_SESSION_ID"
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


def _api_tool_names(api_tools: list[Any]) -> set[str]:
    """Read function names from Chat Completions and Responses tool shapes."""
    names: set[str] = set()
    for tool in api_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        elif tool.get("type") == "function" and isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _uses_responses_tool_shape(api_tools: list[Any]) -> bool:
    """Whether the request is using OpenAI Responses flattened functions."""
    return any(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("name"), str)
        and not isinstance(tool.get("function"), dict)
        for tool in api_tools
    )


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


# --------------------------------------------------------------------------
# 2026-08-16 owner decree: this is the agent's own machine.  Native, internal,
# and optional tools must never be restricted from the model in ANY
# orchestration mode, including a verified Switchboard managed parent.  The
# clamp these functions used to apply caused managed-parent sessions to lose
# local exec entirely and start driving GUI automation tools instead to get
# work done -- worse, not safer.  The functions below are retired in place
# (kept, not deleted, because callers depend on their signatures) and now
# always report "nothing restricted".  Switchboard's managed-parent mode
# remains available as pure metadata/registration plumbing -- orchestration
# mode is still reported in ACP ``_meta`` -- but it has no effect on which
# tools the model can see or call.
# --------------------------------------------------------------------------


def without_switchboard_tool_search_bridge(
    agent: Any,
    tool_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retired 2026-08-16: managed parents keep the full toolset (owner decree).

    This previously restricted a verified managed parent to the Switchboard
    controller tools plus an operator-approved MCP allowlist, stripping
    terminal/code-exec/files/skills. Per the 2026-08-16 owner decree, no
    orchestration mode may restrict tools, so this is now a pure pass-through
    regardless of mode, verification state, or the live tool surface.
    """
    return tool_defs


def _switchboard_parent_tool_names() -> set[str]:
    """Read the discovery-backed approved MCP allowlist without startup coupling."""
    try:
        from tools.mcp_tool import (
            get_switchboard_parent_read_only_tool_names,
            get_switchboard_parent_tool_names,
        )

        # Keep the legacy getter in the union so older plugins and tests that
        # patch it retain their safe, read-only behavior.
        return (
            get_switchboard_parent_read_only_tool_names()
            | get_switchboard_parent_tool_names()
        )
    except ImportError:
        return set()


def _with_switchboard_parent_schemas(
    tool_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Promote explicitly approved MCP schemas deferred by disclosure."""
    selected = _switchboard_parent_tool_names()
    if not selected:
        return tool_defs
    existing = {
        (tool.get("function") or {}).get("name")
        for tool in tool_defs
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    missing = selected - existing
    if not missing:
        return tool_defs
    try:
        from tools.registry import registry
    except ImportError:
        return tool_defs

    promoted = list(tool_defs)
    for name in sorted(missing):
        schema = registry.get_schema(name)
        if isinstance(schema, dict) and schema.get("name") == name:
            promoted.append({"type": "function", "function": dict(schema)})
    return promoted


def _switchboard_visible_tool_names(agent: Any) -> set[str]:
    current = set(_tool_names(agent))
    return set(_SWITCHBOARD_MCP_TOOL_NAMES) | (
        current & _switchboard_parent_tool_names()
    )


def _enforce_verified_switchboard_model_surface(agent: Any) -> None:
    """Retired 2026-08-16: no-op (owner decree, see module header).

    Previously clamped ``agent.tools``/``valid_tool_names`` down to the
    trusted controller surface before ACP attestation. Per the 2026-08-16
    owner decree this must never narrow the model's tool surface, so it is
    now an unconditional no-op; the dead code below is left in place only so
    a future change to ``without_switchboard_tool_search_bridge`` can't
    silently regain teeth through this function.
    """
    return
    current = list(getattr(agent, "tools", None) or [])
    restricted = without_switchboard_tool_search_bridge(agent, current)
    if restricted == current:
        return
    agent.tools = restricted
    agent.valid_tool_names = _tool_names(agent)
    invalidate = getattr(agent, "_invalidate_system_prompt", None)
    if callable(invalidate):
        invalidate()


def restrict_verified_switchboard_request_tools(
    agent: Any,
    api_kwargs: dict[str, Any],
) -> bool | None:
    """Retired 2026-08-16: no-op, always returns None (owner decree).

    Previously clamped a verified managed parent's outgoing provider
    ``tools``/``tool_choice``/``parallel_tool_calls`` down to the trusted
    controller surface, failing closed (denying all tools) when the surface
    didn't exactly match. Per the 2026-08-16 owner decree no orchestration
    mode may restrict or deny tools, so this now always returns ``None`` --
    its own pre-existing "session untouched" contract -- and never mutates
    ``api_kwargs``.
    """
    return None
    if requested_orchestration_mode() != "switchboard":
        return None
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return None

    _enforce_verified_switchboard_model_surface(agent)
    api_tools = api_kwargs.get("tools")
    allowed_names = _switchboard_visible_tool_names(agent)
    allowed_tools: list[Any] = []
    if isinstance(api_tools, list):
        allowed_tools = [
            tool
            for tool in api_tools
            if bool(_api_tool_names([tool]) & allowed_names)
        ]
    if (
        set(_tool_names(agent)) == allowed_names
        and _api_tool_names(allowed_tools) == allowed_names
        and _SWITCHBOARD_MCP_TOOL_NAMES <= allowed_names
    ):
        api_kwargs["tools"] = allowed_tools
        return True

    api_kwargs["tools"] = []
    api_kwargs["tool_choice"] = "none"
    api_kwargs["parallel_tool_calls"] = False
    return False


def apply_switchboard_uat_direct_delegate_once(
    agent: Any,
    api_kwargs: dict[str, Any],
) -> bool:
    """Retired 2026-08-16: no-op, always returns False (owner decree).

    Previously forced one verified managed-parent request's ``tool_choice``
    to the Switchboard delegate controller. Forcing a specific tool choice is
    itself a form of restricting the model's tool surface, which the
    2026-08-16 owner decree forbids in every orchestration mode. This now
    always returns ``False`` -- its own pre-existing "did not force anything"
    outcome -- and never mutates ``api_kwargs``.
    """
    return False
    if os.environ.get(_SWITCHBOARD_FORCE_DIRECT_DELEGATE_ONCE_ENV) != "1":
        return False
    if getattr(agent, "_switchboard_uat_direct_delegate_once_consumed", False):
        return False
    if requested_orchestration_mode() != "switchboard":
        return False
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return False
    if restrict_verified_switchboard_request_tools(agent, api_kwargs) is not True:
        return False

    api_tools = api_kwargs["tools"]

    # Consume before the provider call: retry rebuilding must never make a
    # second logical request forcibly delegate, while a streaming fallback
    # retains these same kwargs for its non-streaming transport retry.
    setattr(agent, "_switchboard_uat_direct_delegate_once_consumed", True)
    if _uses_responses_tool_shape(api_tools):
        api_kwargs["tool_choice"] = {
            "type": "function",
            "name": "mcp__switchboard_orch__delegate",
        }
    else:
        api_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": "mcp__switchboard_orch__delegate"},
        }
    api_kwargs["parallel_tool_calls"] = False
    return True


def switchboard_runtime_tool_block(agent: Any, tool_name: str) -> str | None:
    """Retired 2026-08-16: no-op, always returns None (owner decree).

    Previously denied any non-controller tool call dispatched from a verified
    managed parent. Per the 2026-08-16 owner decree no orchestration mode may
    deny a tool call, so this now always returns ``None`` -- its own
    pre-existing "allowed" contract -- for every mode, verification state,
    and tool name.
    """
    return None
    if requested_orchestration_mode() != "switchboard":
        return None
    if not getattr(agent, "_switchboard_orchestration_mcp_registration_verified", False):
        return None

    switchboard_prefix = f"mcp__{_SWITCHBOARD_MCP_SERVER}__"
    effective_switchboard_tools = {
        name for name in _tool_names(agent) if name.startswith(switchboard_prefix)
    }
    if effective_switchboard_tools != _SWITCHBOARD_MCP_TOOL_NAMES:
        return None

    if tool_name in _switchboard_visible_tool_names(agent):
        return None
    return (
        "Switchboard orchestration runtime policy permits only the verified "
        "controller tools and explicitly approved MCP tools: "
        + ", ".join(sorted(_switchboard_visible_tool_names(agent)))
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

    trusted_child_env: dict[str, str] = {}
    for env_name in (_SWITCHBOARD_GATEWAY_URL_ENV, _SWITCHBOARD_SESSION_ID_ENV):
        value = (os.environ.get(env_name) or "").strip()
        if not value:
            raise RuntimeError(f"{env_name} is required")
        trusted_child_env[env_name] = value

    hardened = dict(config)
    hardened["command"] = expected_command
    hardened["args"] = []
    # The ACP request is forbidden from supplying environment variables, but
    # Hermes' stdio launcher uses a scrubbed environment rather than inheriting
    # the ACP process wholesale. Forward only the two session-scoped values
    # that Switchboard placed on the trusted parent process. Without this
    # explicit allowlist the shim receives an empty base URL and every native
    # provider reports reqwest's opaque "builder error" before reaching the
    # gateway.
    hardened["env"] = trusted_child_env
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

    _enforce_verified_switchboard_model_surface(agent)
    disabled = sorted(set(getattr(agent, "disabled_toolsets", None) or []))
    tools = _tool_names(agent)
    mcp_servers = _mcp_server_names(agent)
    has_native_delegate = "delegate_task" in tools
    switchboard_tool_prefix = f"mcp__{_SWITCHBOARD_MCP_SERVER}__"
    switchboard_tools = {
        name for name in tools if name.startswith(switchboard_tool_prefix)
    }
    exact_switchboard_tools = switchboard_tools == _SWITCHBOARD_MCP_TOOL_NAMES
    trusted_switchboard_registration = bool(
        getattr(
            agent,
            "_switchboard_orchestration_mcp_registration_verified",
            False,
        )
    )
    if (
        trusted_switchboard_registration
        and switchboard_tools == _SWITCHBOARD_MCP_TOOL_NAMES
        and _SWITCHBOARD_MCP_SERVER not in mcp_servers
    ):
        # A late refresh may drop the bookkeeping entry while preserving the
        # connected, attested session server and its exact live tool surface.
        mcp_servers.append(_SWITCHBOARD_MCP_SERVER)
        mcp_servers.sort()
    has_switchboard_server = _SWITCHBOARD_MCP_SERVER in mcp_servers
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
