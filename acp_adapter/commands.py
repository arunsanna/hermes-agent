"""Headless slash commands for ACP sessions (``/help``, ``/model``, ``/compress`` ...)."""

from __future__ import annotations

import contextvars
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from acp.schema import AvailableCommand, AvailableCommandsUpdate, UnstructuredCommandInput

from acp_adapter.session import SessionState, _expand_acp_enabled_toolsets
from hermes_cli.commands import resolve_command

logger = logging.getLogger("acp_adapter.server")

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"


def _hint_from_args_hint(args_hint: str) -> str | None:
    """Strip a CommandDef's surrounding ``[ ]``/``< >`` wrapper; empty -> None."""
    return args_hint.strip("[]<>").strip() or None


# Registry wording that doesn't hold for ACP's simpler handlers: name -> (description, hint).
# Everything not listed here is derived straight from hermes_cli.commands (see _build_commands).
ACP_COMMAND_OVERRIDES: dict[str, tuple[str, str | None]] = {
    # Registry documents "/help skills" (skill-command listing) and "/help <filter>" (text
    # search); ACP's _cmd_help ignores args entirely and always lists every command.
    "help": ("List available commands", None),
    # Registry documents --provider/--global/--session/--refresh flags; ACP's _cmd_model only
    # ever accepts a bare model name and switches it for this session.
    "model": ("Show current model and provider, or switch models", "model name to switch to"),
    # Registry's /tools is cli_only with list/disable/enable subcommands; ACP's _cmd_tools
    # ignores args and always lists every enabled tool.
    "tools": ("List available tools with descriptions", None),
    # Registry documents a "context all" subcommand and throughput stats; ACP's _cmd_context
    # ignores args and reports no throughput figure.
    "context": ("Show conversation context info", None),
    # Registry entry is "new" (reset's alias) and describes starting a brand-new session id;
    # ACP's _cmd_reset just clears history in place, same session id, and takes no name arg.
    "reset": ("Clear conversation history", None),
    # Registry documents "here [N]" partial compression and a --preview/--dry-run mode; ACP's
    # _cmd_compress always compresses the full history for real, no partial/preview support.
    "compress": ("Compress conversation context", None),
    # The CLI/gateway registry only has a *management* command for skills as a whole ("skills":
    # search/install/inspect/...); loading ONE skill by name has no static CommandDef — it's a
    # dynamically-registered "/<skill-name>" command per skill (agent/skill_commands.py). ACP
    # exposes that capability explicitly as "/skill <name>", so this entry has no registry
    # counterpart to derive from at all (see REGISTRY_LESS_OVERRIDES in test_server.py).
    "skill": ("Load a skill's instructions into this session", "name [instruction]"),
    # Registry describes "Show session, model, token, and context info"; ACP's _cmd_status
    # only reports session id, model, provider, and per-role message counts -- no token count
    # or context-window figure (that promise belongs to /context, which does report it).
    "status": ("Show session, model, and message counts", None),
}


def _build_commands() -> dict[str, tuple[str, str | None]]:
    """The executable command set, name -> (advertised description, input hint).

    Derived from hermes_cli.commands.resolve_command() so the ACP surface stays in sync with
    the CLI/gateway registry, except for ACP_COMMAND_OVERRIDES entries whose registry wording
    describes CLI/gateway-only behavior the ACP handlers don't implement (or, for "skill", has
    no registry entry to derive from at all).
    """
    commands: dict[str, tuple[str, str | None]] = {}
    for name in ("help", "model", "tools", "context", "reset", "compress", "steer", "queue", "version",
                 "goal", "subgoal", "status", "skill"):
        if name in ACP_COMMAND_OVERRIDES:
            commands[name] = ACP_COMMAND_OVERRIDES[name]
            continue
        registry_def = resolve_command(name)
        if registry_def is None:
            raise RuntimeError(f"hermes_cli.commands registry has no entry for /{name}")
        commands[name] = (registry_def.description, _hint_from_args_hint(registry_def.args_hint))
    return commands


@dataclass(frozen=True)
class RunPromptAfterCommand:
    """Sentinel a handler returns to emit ``notice`` as agent text, then run a normal turn with
    ``prompt_text`` as the user's next message (e.g. ``/goal <text>``, ``/goal resume``, ``/skill
    <name>``) — as opposed to a plain ``str`` return, which ends the turn with no further work."""

    notice: str
    prompt_text: str


def _estimate_tokens(history: list, agent: Any, system_prompt: str | None = None, tools: Any = None) -> int:
    """Rough request-token estimate over history + system prompt + tool schemas."""
    from agent.model_metadata import estimate_request_tokens_rough

    if system_prompt is None:
        system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    if tools is None:
        tools = getattr(agent, "tools", None) or None
    return estimate_request_tokens_rough(history, system_prompt=system_prompt, tools=tools)


def _queue_prompt(state: SessionState, text: str) -> int:
    with state.runtime_lock:
        state.queued_prompts.append(text)
        return len(state.queued_prompts)


def _get_goal_manager(state: SessionState) -> Any:
    """Lazily construct (and cache on the session) the GoalManager bound to this ACP session's
    persisted goal state — the ACP analogue of the CLI's ``_get_goal_manager`` (hermes_cli/
    cli_loops_mixin.py) and the gateway's ``_get_goal_manager_for_event`` (gateway/run_goals.py).

    ``GoalManager.__init__`` loads any persisted ``goal:<session_id>`` state itself, so caching
    the instance on ``state.goal_manager`` (keyed by the same ACP session id across load/resume)
    is enough to keep this session's goal in sync with what ``/goal`` last set.
    """
    if state.goal_manager is None:
        from hermes_cli.config import load_config
        from hermes_cli.goals import GoalManager

        try:
            goals_cfg = (load_config() or {}).get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        state.goal_manager = GoalManager(session_id=state.session_id, default_max_turns=max_turns)
    return state.goal_manager


class SlashCommandsMixin:
    """Slash-command surface for ``HermesACPAgent``; relies on ``_conn``, ``_send``, ``_schedule_soon``,
    ``session_manager`` and ``_switch_model`` from the host class."""

    # name -> (advertised description, input hint); see _build_commands / ACP_COMMAND_OVERRIDES.
    _COMMANDS: dict[str, tuple[str, str | None]] = _build_commands()

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        return [
            AvailableCommand(name=name, description=desc, input=UnstructuredCommandInput(hint=hint) if hint else None)
            for name, (desc, hint) in cls._COMMANDS.items()
        ]

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return
        update = AvailableCommandsUpdate(
            session_update="available_commands_update", available_commands=self._available_commands()
        )
        await self._send(session_id, update, fail_msg="Failed to advertise ACP slash commands for session %s")

    def _schedule_available_commands_update(self, session_id: str) -> None:
        self._schedule_soon(lambda: self._send_available_commands_update(session_id))

    def _handle_slash_command(self, text: str, state: SessionState) -> str | RunPromptAfterCommand | None:
        """Dispatch a slash command; ``None`` for unknown ones so they fall through to the LLM.

        A handler returns ``RunPromptAfterCommand`` instead of ``str`` when it wants the caller
        (``prompt()``) to emit a notice and then run a normal turn (e.g. ``/goal <text>``).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd not in self._COMMANDS:
            return None
        handler = getattr(self, f"_cmd_{cmd}")

        # Handlers run on the loop thread, outside the per-turn cwd-pinning context. ``/compress``
        # and ``/model`` REBUILD the system prompt, so unpinned they'd bake the Hermes install tree
        # into the persisted cached prompt. Pin inside a fresh context: no leak, no teardown.
        def _dispatch() -> str | RunPromptAfterCommand | None:
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
        lines.extend(f"  /{cmd:10s}  {desc}" for cmd, (desc, _hint) in self._COMMANDS.items())
        lines.extend(["", "Unrecognized /commands are sent to the model as normal messages."])
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider, target_provider, new_model = self._switch_model(state, args)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider or "openrouter"
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            from types import SimpleNamespace
            from agent.memory_manager import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"])
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={t.get("function", {}).get("name") for t in tools or [] if isinstance(t, dict)},
                enabled_toolsets=toolsets, _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = (t.get("function") or {}).get("name", "?")
                desc = (t.get("function") or {}).get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)
        roles = Counter(msg.get("role", "unknown") for msg in state.history)

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            approx_tokens = _estimate_tokens(state.history, agent)
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages" if n_messages else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0 and context_length > 0:
            usage_pct = (approx_tokens / context_length) * 100
            lines.append(f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)")
        elif approx_tokens > 0:
            lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0 and approx_tokens > 0:
            threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
            pct_note = f", {threshold_pct:.0f}%" if threshold_pct else ""
            if approx_tokens >= threshold_tokens:
                lines.append(f"Compression: due now (threshold ~{threshold_tokens:,}{pct_note}). Run /compress.")
            else:
                remaining = max(threshold_tokens - approx_tokens, 0)
                lines.append(f"Compression: ~{remaining:,} tokens until threshold (~{threshold_tokens:,}{pct_note}).")
        elif threshold_tokens > 0:
            lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        lines.append(
            "Auto-compaction is disabled (compression.enabled: false); /compress still compresses manually."
            if getattr(agent, "compression_enabled", True) is False
            else "Tip: run /compress to compress manually before the threshold."
        )
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        try:
            reset_session_state = getattr(state.agent, "reset_session_state", None)
            if callable(reset_session_state):
                reset_session_state()
        except Exception:
            logger.warning("ACP session state reset failed for %s", state.session_id, exc_info=True)
            return "Conversation history cleared. Agent session state reset failed; see logs."
        finally:
            self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compress(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            # No compression_enabled gate: it only disables *automatic* compaction (CLI/gateway parity).
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real request pressure.
            # See #6217.
            # See #6217.
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = _estimate_tokens(state.history, agent, _sys_prompt, _tools)
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # Stable ACP session id: suppress _compress_context's SQLite session split.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history, _sys_prompt, approx_tokens=approx_tokens, task_id=state.session_id, force=True,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_tokens = _estimate_tokens(
                state.history, agent, getattr(agent, "_cached_system_prompt", "") or _sys_prompt,
                getattr(agent, "tools", None) or _tools,
            )
            return (
                f"Context compressed: {original_count} -> {len(state.history)} messages\n"
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

        return f"No active turn — queued for the next turn. ({_queue_prompt(state, steer_text)} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        return f"Queued for the next turn. ({_queue_prompt(state, queued_text)} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- /goal, /subgoal, /status, /skill --------------------------------------------------
    # Verb table mirrors hermes_cli.cli_commands_mixin._handle_goal_command /
    # gateway.slash_commands_goals._handle_goal_command exactly (same GoalManager, same
    # persisted state, same notice wording); only the dispatch surface differs (return values
    # instead of printing / posting).

    def _cmd_goal(self, args: str, state: SessionState) -> str | RunPromptAfterCommand:
        mgr = _get_goal_manager(state)
        lower = args.lower()
        verb, _, rest = args.partition(" ")
        verb = verb.lower()
        rest = rest.strip()

        if not args or lower == "status":
            return mgr.status_line()
        if lower == "show":
            return f"{mgr.status_line()}\n{mgr.render_contract()}"
        if lower.startswith("draft"):
            objective = args[len("draft"):].strip()
            if not objective:
                return "Usage: /goal draft <objective in plain language>"
            return self._goal_draft(mgr, objective)
        if lower == "pause":
            goal_state = mgr.pause(reason="user-paused")
            return f"⏸ Goal paused: {goal_state.goal}" if goal_state else "No goal set."
        if lower == "resume":
            return self._goal_resume(mgr)
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return "✓ Goal cleared." if had else "No active goal."
        if verb == "wait":
            return self._goal_wait(mgr, rest)
        if lower == "unwait":
            return "▶ Wait barrier cleared — goal loop resumes." if mgr.stop_waiting() else "No wait barrier set."
        if verb == "gate":
            return self._goal_gate(mgr, rest)
        return self._goal_set(mgr, args)

    def _goal_set_notice(self, goal_state: Any, contract_label: str) -> str:
        lines = [f"⊙ Goal set ({goal_state.max_turns}-turn budget): {goal_state.goal}"]
        if goal_state.has_contract():
            lines.append(contract_label)
            lines.extend(f"  {line}" for line in goal_state.contract.render_block().splitlines())
        against = " against the contract above" if goal_state.has_contract() else ""
        lines.append(
            f"After each turn, a judge model checks if the goal is done{against}. Hermes keeps "
            "working until it is, you pause/clear it, or the budget is exhausted. Use /goal "
            "status, /goal show, /goal pause, /goal resume, /goal clear."
        )
        return "\n".join(lines)

    def _goal_set(self, mgr: Any, arg: str) -> str | RunPromptAfterCommand:
        from hermes_cli.goals import parse_contract

        headline, contract = parse_contract(arg)
        try:
            goal_state = mgr.set(headline or arg, contract=contract if not contract.is_empty() else None)
        except ValueError as e:
            return f"Invalid goal: {e}"
        notice = self._goal_set_notice(goal_state, "Completion contract:")
        return RunPromptAfterCommand(notice=notice, prompt_text=goal_state.goal)

    def _goal_draft(self, mgr: Any, objective: str) -> str | RunPromptAfterCommand:
        from hermes_cli.goals import draft_contract

        try:
            contract = draft_contract(objective)
        except Exception:
            logger.debug("ACP goal draft failed", exc_info=True)
            contract = None
        try:
            goal_state = mgr.set(objective, contract=contract)
        except ValueError as e:
            return f"Invalid goal: {e}"
        notice = self._goal_set_notice(goal_state, "Drafted completion contract:")
        if not goal_state.has_contract():
            notice += (
                "\nCouldn't draft a contract (aux model unavailable) — running as a free-form "
                "goal. The per-turn judge still applies."
            )
        return RunPromptAfterCommand(notice=notice, prompt_text=goal_state.goal)

    def _goal_resume(self, mgr: Any) -> str | RunPromptAfterCommand:
        goal_state = mgr.resume()
        if goal_state is None:
            return "No goal to resume."
        notice = f"▶ Goal resumed: {goal_state.goal}"
        prompt = mgr.next_continuation_prompt()
        if prompt:
            return RunPromptAfterCommand(notice=notice, prompt_text=prompt)
        return f"{notice}\nSend any message to kick off the next step."

    def _goal_wait(self, mgr: Any, wait_arg: str) -> str:
        if not wait_arg:
            return "Usage: /goal wait <pid> [reason]"
        tokens = wait_arg.split(None, 1)
        try:
            pid = int(tokens[0])
        except ValueError:
            return "/goal wait: <pid> must be an integer process id."
        reason = tokens[1].strip() if len(tokens) > 1 else ""
        try:
            mgr.wait_on(pid, reason=reason)
        except (RuntimeError, ValueError) as e:
            return f"/goal wait: {e}"
        rtxt = f" ({reason})" if reason else ""
        return f"⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits."

    def _goal_gate(self, mgr: Any, gate_arg: str) -> str:
        gate_lower = gate_arg.lower()
        if not gate_arg or gate_lower == "list":
            return mgr.render_gates()
        if gate_lower.startswith("add "):
            try:
                gate = mgr.add_gate(gate_arg[len("add"):].strip())
            except (RuntimeError, ValueError) as e:
                return f"/goal gate add: {e}"
            return (
                f"⚿ Gate added: $ {gate.command} ({gate.max_retries} retries, "
                f"{gate.timeout_seconds}s timeout). It must pass before the goal can complete."
            )
        if gate_lower.startswith("remove ") or gate_lower.startswith("rm "):
            try:
                idx = int(gate_arg.split(None, 1)[1].strip())
                removed = mgr.remove_gate(idx)
            except (RuntimeError, ValueError, IndexError) as e:
                return f"/goal gate remove: {e}"
            return f"✓ Gate removed: $ {removed}"
        if gate_lower == "clear":
            try:
                prev = mgr.clear_gates()
            except RuntimeError as e:
                return f"/goal gate clear: {e}"
            return f"✓ Cleared {prev} gate{'s' if prev != 1 else ''}."
        return "Usage: /goal gate [list | add <command> | remove <N> | clear]"

    def _cmd_subgoal(self, args: str, state: SessionState) -> str:
        mgr = _get_goal_manager(state)
        arg = args.strip()
        if not arg:
            return mgr.render_subgoals()
        lower = arg.lower()
        if lower == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as e:
                return f"/subgoal clear: {e}"
            return f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}." if prev else "No subgoals to clear."
        if lower.startswith("remove "):
            rest = arg[len("remove"):].strip()
            if not rest:
                return "Usage: /subgoal remove <n>"
            try:
                idx = int(rest)
            except ValueError:
                return "/subgoal remove: <n> must be an integer (1-based index)."
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as e:
                return f"/subgoal remove: {e}"
            return f"✓ Removed subgoal {idx}: {removed}"
        try:
            text = mgr.add_subgoal(arg)
        except (ValueError, RuntimeError) as e:
            return f"/subgoal: {e}"
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return f"✓ Added subgoal {idx}: {text}"

    def _cmd_status(self, args: str, state: SessionState) -> str:
        model = state.model or getattr(state.agent, "model", "") or "unknown"
        provider = getattr(state.agent, "provider", None) or "auto"
        roles = Counter(msg.get("role", "unknown") for msg in state.history)
        lines = [
            f"Session: {state.session_id}",
            f"Model: {model}",
            f"Provider: {provider}",
            f"Messages: {len(state.history)} (user: {roles.get('user', 0)}, "
            f"assistant: {roles.get('assistant', 0)}, tool: {roles.get('tool', 0)}, "
            f"system: {roles.get('system', 0)})",
        ]
        mgr = _get_goal_manager(state)
        if mgr.has_goal():
            lines.append(mgr.status_line())
        return "\n".join(lines)

    def _cmd_skill(self, args: str, state: SessionState) -> str | RunPromptAfterCommand:
        parts = args.strip().split(None, 1)
        if not parts:
            return "Usage: /skill <name> [instruction]"
        name = parts[0].lstrip("/")
        instruction = parts[1].strip() if len(parts) > 1 else ""

        from agent.skill_commands import build_skill_invocation_message, get_skill_commands

        cmd_key = f"/{name}"
        skill_info = get_skill_commands().get(cmd_key)
        if not skill_info:
            return f"Unknown skill: {name}"
        msg = build_skill_invocation_message(cmd_key, instruction, task_id=state.session_id)
        if not msg:
            return f"Failed to load skill: {name}"
        return RunPromptAfterCommand(notice=f"⚡ Loading skill: {skill_info['name']}", prompt_text=msg)
