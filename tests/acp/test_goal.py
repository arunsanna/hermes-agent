"""Tests for the ACP /goal, /subgoal, /status, /skill slash commands (Phase 3a)."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import TextContentBlock

from acp_adapter.commands import ACP_COMMAND_OVERRIDES, RunPromptAfterCommand
from hermes_cli.goals import load_goal

# Reuse the ACP fixtures from test_server.py instead of redefining them.
from tests.acp.test_server import agent, mock_manager  # noqa: F401


def _make_state(mock_manager):
    state = mock_manager.create_session(cwd="/tmp")
    state.agent.model = "test-model"
    state.agent.provider = "openrouter"
    state.model = "test-model"
    return state


# ---------------------------------------------------------------------------
# Advertising (Phase 1 contract, extended to the new commands)
# ---------------------------------------------------------------------------


class TestGoalCommandsAdvertised:
    def test_new_commands_are_advertised(self, agent):
        names = {cmd.name for cmd in agent._available_commands()}
        assert {"goal", "subgoal", "status", "skill"} <= names

    def test_skill_is_a_registry_less_override(self):
        assert "skill" in ACP_COMMAND_OVERRIDES


# ---------------------------------------------------------------------------
# /goal — unit-level verb dispatch
# ---------------------------------------------------------------------------


class TestGoalCommand:
    def test_goal_status_with_no_goal(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/goal status", state)
        assert isinstance(result, str)
        assert "No active goal" in result

    def test_goal_bare_same_as_status(self, agent, mock_manager):
        state = _make_state(mock_manager)
        assert agent._handle_slash_command("/goal", state) == agent._handle_slash_command("/goal status", state)

    def test_goal_set_returns_sentinel_and_persists(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/goal Ship the widget", state)

        assert isinstance(result, RunPromptAfterCommand)
        assert "⊙ Goal set" in result.notice
        assert "Ship the widget" in result.notice
        assert result.prompt_text == "Ship the widget"

        goal_state = load_goal(state.session_id)
        assert goal_state is not None
        assert goal_state.goal == "Ship the widget"
        assert goal_state.status == "active"

    def test_goal_show_includes_status_and_contract(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        result = agent._handle_slash_command("/goal show", state)
        assert "Ship the widget" in result
        assert "no completion contract" in result.lower()

    def test_goal_pause_and_status_reflect_it(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        result = agent._handle_slash_command("/goal pause", state)
        assert "paused" in result.lower()
        assert load_goal(state.session_id).status == "paused"
        assert "paused" in agent._handle_slash_command("/goal status", state).lower()

    def test_goal_resume_returns_sentinel_and_reactivates(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        agent._handle_slash_command("/goal pause", state)

        result = agent._handle_slash_command("/goal resume", state)

        assert isinstance(result, RunPromptAfterCommand)
        assert "resumed" in result.notice.lower()
        assert result.prompt_text  # a continuation prompt was enqueued
        assert load_goal(state.session_id).status == "active"

    def test_goal_resume_with_no_goal(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/goal resume", state)
        assert result == "No goal to resume."

    def test_goal_clear(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        result = agent._handle_slash_command("/goal clear", state)
        assert "cleared" in result.lower()
        assert load_goal(state.session_id).status == "cleared"

    def test_goal_clear_with_no_goal(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/goal clear", state)
        assert "no active goal" in result.lower()

    def test_goal_gate_add_list_remove_clear(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)

        added = agent._handle_slash_command("/goal gate add pytest -q", state)
        assert "Gate added" in added
        assert "pytest -q" in added

        listed = agent._handle_slash_command("/goal gate list", state)
        assert "pytest -q" in listed

        removed = agent._handle_slash_command("/goal gate remove 1", state)
        assert "Gate removed" in removed

        cleared = agent._handle_slash_command("/goal gate clear", state)
        assert "Cleared 0 gate" in cleared  # already removed above

    def test_goal_wait_and_unwait(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)

        wait_result = agent._handle_slash_command(f"/goal wait {os.getpid()} building", state)
        assert "parked on pid" in wait_result.lower()
        assert "parked" in agent._handle_slash_command("/goal status", state).lower()

        unwait_result = agent._handle_slash_command("/goal unwait", state)
        assert "cleared" in unwait_result.lower()

    def test_goal_wait_requires_integer_pid(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        result = agent._handle_slash_command("/goal wait not-a-pid", state)
        assert "integer" in result.lower()

    def test_goal_draft_uses_drafted_contract(self, agent, mock_manager):
        from hermes_cli.goals import GoalContract

        state = _make_state(mock_manager)
        contract = GoalContract(outcome="ship it", verification="tests pass")
        with patch("hermes_cli.goals.draft_contract", return_value=contract) as mock_draft:
            result = agent._handle_slash_command("/goal draft Ship the widget well", state)

        mock_draft.assert_called_once_with("Ship the widget well")
        assert isinstance(result, RunPromptAfterCommand)
        assert "Drafted completion contract" in result.notice
        goal_state = load_goal(state.session_id)
        assert goal_state.contract.outcome == "ship it"

    def test_goal_draft_falls_back_when_judge_unreachable(self, agent, mock_manager):
        """draft_contract returning None (aux model unavailable) must not block goal-setting."""
        state = _make_state(mock_manager)
        with patch("hermes_cli.goals.draft_contract", return_value=None):
            result = agent._handle_slash_command("/goal draft Ship the widget well", state)

        assert isinstance(result, RunPromptAfterCommand)
        assert "aux model unavailable" in result.notice.lower()
        goal_state = load_goal(state.session_id)
        assert goal_state.goal == "Ship the widget well"
        assert not goal_state.has_contract()

    def test_goal_draft_never_raises_on_judge_exception(self, agent, mock_manager):
        """A network/API failure inside draft_contract must degrade to a plain goal, not raise."""
        state = _make_state(mock_manager)
        with patch("hermes_cli.goals.draft_contract", side_effect=RuntimeError("network down")):
            result = agent._handle_slash_command("/goal draft Ship the widget well", state)

        assert isinstance(result, RunPromptAfterCommand)
        goal_state = load_goal(state.session_id)
        assert goal_state.goal == "Ship the widget well"

    def test_goal_draft_without_objective(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/goal draft", state)
        assert "Usage: /goal draft" in result


# ---------------------------------------------------------------------------
# /subgoal
# ---------------------------------------------------------------------------


class TestSubgoalCommand:
    def test_subgoal_bare_with_no_goal(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/subgoal", state)
        assert "no active goal" in result.lower()

    def test_subgoal_add_list_remove_clear(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)

        add1 = agent._handle_slash_command("/subgoal Write tests", state)
        assert "Added subgoal 1: Write tests" in add1
        add2 = agent._handle_slash_command("/subgoal Update docs", state)
        assert "Added subgoal 2: Update docs" in add2

        listing = agent._handle_slash_command("/subgoal", state)
        assert "Write tests" in listing
        assert "Update docs" in listing

        removed = agent._handle_slash_command("/subgoal remove 1", state)
        assert "Write tests" in removed

        cleared = agent._handle_slash_command("/subgoal clear", state)
        assert "Cleared 1 subgoal" in cleared

        assert "no subgoals" in agent._handle_slash_command("/subgoal", state).lower()


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_reports_session_and_model(self, agent, mock_manager):
        state = _make_state(mock_manager)
        state.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        result = agent._handle_slash_command("/status", state)
        assert state.session_id in result
        assert "test-model" in result
        assert "Messages: 2" in result

    def test_status_includes_goal_line_when_active(self, agent, mock_manager):
        state = _make_state(mock_manager)
        agent._handle_slash_command("/goal Ship the widget", state)
        result = agent._handle_slash_command("/status", state)
        assert "Ship the widget" in result

    def test_status_advertised_description_does_not_overpromise_tokens_or_context(self, agent):
        """Regression: /status is advertised (available_commands_update) with a
        description that must match what _cmd_status actually reports.
        _cmd_status never computes a token count or context-window figure, so
        the advertised wording must not promise one (that promise belongs to
        /context, which does report it).
        """
        descriptions = {cmd.name: cmd.description for cmd in agent._available_commands()}
        assert "token" not in descriptions["status"].lower()
        assert "context" not in descriptions["status"].lower()


# ---------------------------------------------------------------------------
# /skill
# ---------------------------------------------------------------------------


class TestSkillCommand:
    def test_skill_loads_known_skill(self, agent, mock_manager):
        state = _make_state(mock_manager)
        skill_info = {"name": "My Skill", "description": "does things",
                      "skill_md_path": "/skills/my-skill/SKILL.md", "skill_dir": "/skills/my-skill"}
        with (
            patch("agent.skill_commands.get_skill_commands", return_value={"/my-skill": skill_info}),
            patch("agent.skill_commands.build_skill_invocation_message", return_value="SKILL BODY") as mock_build,
        ):
            result = agent._handle_slash_command("/skill my-skill do the thing", state)

        assert isinstance(result, RunPromptAfterCommand)
        assert "My Skill" in result.notice
        assert result.prompt_text == "SKILL BODY"
        mock_build.assert_called_once_with("/my-skill", "do the thing", task_id=state.session_id)

    def test_skill_unknown_name_returns_error(self, agent, mock_manager):
        state = _make_state(mock_manager)
        with patch("agent.skill_commands.get_skill_commands", return_value={}):
            result = agent._handle_slash_command("/skill nonexistent-skill", state)
        assert isinstance(result, str)
        assert "Unknown skill" in result

    def test_skill_no_args_shows_usage(self, agent, mock_manager):
        state = _make_state(mock_manager)
        result = agent._handle_slash_command("/skill", state)
        assert "Usage: /skill" in result


# ---------------------------------------------------------------------------
# prompt() integration — the sentinel must run a normal turn, plain strings must not
# ---------------------------------------------------------------------------


class TestGoalPromptIntegration:
    @pytest.mark.asyncio
    async def test_goal_status_with_no_goal_does_not_run_agent(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "should not run", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="/goal status")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_not_called()
        updates = [c.kwargs.get("update") or c.args[1] for c in mock_conn.session_update.call_args_list]
        texts = [u.content.text for u in updates if u.session_update == "agent_message_chunk"]
        assert any("No active goal" in t for t in texts)

    @pytest.mark.asyncio
    async def test_goal_set_runs_a_turn_with_the_goal_text(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "ok", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        # Phase 3b wires the goal-continuation loop into every normal turn completion
        # while a goal is active; patch judge_goal so this single-turn test doesn't
        # hit the (unconfigured, network-reaching) real judge and doesn't continue.
        with patch("hermes_cli.goals.judge_goal", return_value=("done", "not evaluated here", False, None, False)):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()
        _, kwargs = state.agent.run_conversation.call_args
        assert kwargs["user_message"] == "Ship the widget"

        goal_state = load_goal(new_resp.session_id)
        assert goal_state is not None
        assert goal_state.goal == "Ship the widget"

        updates = [c.kwargs.get("update") or c.args[1] for c in mock_conn.session_update.call_args_list]
        texts = [u.content.text for u in updates if u.session_update == "agent_message_chunk"]
        assert any("⊙ Goal set" in t for t in texts)
        assert any(t == "ok" for t in texts)


# ---------------------------------------------------------------------------
# Phase 3b — the judge-driven continuation loop inside one prompt() call
# ---------------------------------------------------------------------------


def _agent_message_texts(mock_conn):
    updates = [c.kwargs.get("update") or c.args[1] for c in mock_conn.session_update.call_args_list]
    return [u.content.text for u in updates if u.session_update == "agent_message_chunk"]


class TestGoalContinuationLoop:
    @pytest.mark.asyncio
    async def test_continues_once_then_completes_in_one_prompt_call(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(
            side_effect=[
                {"final_response": "did step 1", "messages": []},
                {"final_response": "did step 2, done", "messages": []},
            ]
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        verdicts = iter([("continue", "keep going", False, None, False), ("done", "shipped", False, None, False)])

        def fake_judge(goal, last_response, **kwargs):
            return next(verdicts)

        with patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 2

        texts = _agent_message_texts(mock_conn)
        continuing = next(t for t in texts if t.startswith("↻ Continuing toward goal"))
        done_notice = next(t for t in texts if t.startswith("✓ Goal achieved"))
        assert (
            texts.index("did step 1")
            < texts.index(continuing)
            < texts.index("did step 2, done")
            < texts.index(done_notice)
        )

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "done"

    @pytest.mark.asyncio
    async def test_stops_at_turn_budget_with_paused_notice(self, agent):
        from hermes_cli.goals import GoalManager

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.goal_manager = GoalManager(session_id=new_resp.session_id, default_max_turns=2)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "still working", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "not done yet", False, None, False)):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 2

        texts = _agent_message_texts(mock_conn)
        assert any(t.startswith("⏸ Goal paused") and "2/2 turns used" in t for t in texts)

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"

    @pytest.mark.asyncio
    async def test_stops_when_cancel_races_in_during_the_judge_call(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "step 1 done", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        def fake_judge(goal, last_response, **kwargs):
            # Simulate a session/cancel landing while the (slow, real) judge call
            # is in flight for this turn's post-completion evaluation.
            state.cancel_event.set()
            return ("continue", "keep going", False, None, False)

        with patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"
        assert state.agent.run_conversation.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_during_judge_pauses_goal_and_suppresses_continue_notice(self, agent):
        """Regression: a session/cancel that lands while the judge is deliberating
        must pause the goal with the CLI-parity reason/notice instead of leaving
        it active, and the "Continuing toward goal" notice for the judge's
        continue decision must never be streamed once cancel is set.
        """
        from acp.schema import PromptResponse

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "step 1 done", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        def fake_judge(goal, last_response, **kwargs):
            # Simulate a session/cancel landing while the (slow, real) judge call
            # is "deliberating" for this turn's post-completion evaluation.
            state.cancel_event.set()
            return ("continue", "keep going", False, None, False)

        with patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "cancelled"
        assert state.agent.run_conversation.call_count == 1

        texts = _agent_message_texts(mock_conn)
        assert not any(t.startswith("↻ Continuing toward goal") for t in texts)
        assert any(
            t == "⏸ Goal paused — turn was interrupted. "
            "Use /goal resume to continue, or /goal clear to stop."
            for t in texts
        )

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"
        assert goal_state.paused_reason == "user-interrupted (Ctrl+C)"

    @pytest.mark.asyncio
    async def test_cancel_before_judge_pauses_goal(self, agent):
        """Regression: a cancel that lands after the turn already committed to
        "final" (e.g. during the auto-title step that runs between the
        terminal-winner commit and the goal-continuation block) but before
        that block's own cancel check must pause the goal with the
        CLI-parity notice, and the judge must never be consulted.
        """
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "step 1 done", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        def fake_auto_title(*args, **kwargs):
            # Simulate a session/cancel landing between this turn's
            # terminal-winner commit (already "final") and the
            # goal-continuation block's own cancel check a beat later.
            state.cancel_event.set()

        with (
            patch("agent.title_generator.maybe_auto_title", side_effect=fake_auto_title),
            patch("hermes_cli.goals.judge_goal") as mock_judge,
        ):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"
        assert state.agent.run_conversation.call_count == 1
        mock_judge.assert_not_called()

        texts = _agent_message_texts(mock_conn)
        assert any(
            t == "⏸ Goal paused — turn was interrupted. "
            "Use /goal resume to continue, or /goal clear to stop."
            for t in texts
        )

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"
        assert goal_state.paused_reason == "user-interrupted (Ctrl+C)"

    @pytest.mark.asyncio
    async def test_stops_when_goal_paused_mid_loop(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        calls = {"n": 0}

        def run_conversation_side_effect(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                # Simulate a concurrent /goal pause command landing between
                # this loop's first continuation and the second turn's own
                # post-completion check.
                state.goal_manager.pause(reason="test-pause")
            return {"final_response": f"step {calls['n']} done", "messages": []}

        state.agent.run_conversation = MagicMock(side_effect=run_conversation_side_effect)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch(
            "hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)
        ) as mock_judge:
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 2
        # The 2nd turn's own post-completion check finds the goal already paused
        # and never consults the judge a second time.
        assert mock_judge.call_count == 1

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"

    @pytest.mark.asyncio
    async def test_stops_for_a_queued_user_prompt_and_lets_the_drain_path_run(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def run_conversation_side_effect(**kwargs):
            if state.agent.run_conversation.call_count == 1:
                # Simulate a second, real prompt() call queuing while this
                # turn was in flight.
                state.queued_prompts.append("wait, do X instead")
            return {"final_response": "step done", "messages": []}

        state.agent.run_conversation = MagicMock(side_effect=run_conversation_side_effect)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch(
            "hermes_cli.goals.judge_goal", return_value=("done", "shipped", False, None, False)
        ) as mock_judge:
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 2
        # Only the drained (queued) turn's own post-completion check ever
        # consults the judge -- the original loop saw the queue was non-empty
        # and stopped continuing before calling it at all.
        assert mock_judge.call_count == 1

        first_args, first_kwargs = state.agent.run_conversation.call_args_list[0]
        second_args, second_kwargs = state.agent.run_conversation.call_args_list[1]
        assert first_kwargs["user_message"] == "Ship the widget"
        assert second_kwargs["user_message"] == "wait, do X instead"

    @pytest.mark.asyncio
    async def test_a_real_cancel_during_the_judge_call_is_not_dropped(self, agent):
        """Regression: state.turn_terminal_winner is committed to "final" at the
        end of each turn and must not stay stale into the next continuation
        iteration's judge deliberation -- otherwise a *real* session/cancel
        landing during that (possibly slow) call hits cancel()'s
        finalized-session guard (server.py's cancel()) and is silently
        dropped, letting the loop keep running to completion/budget instead
        of stopping. Unlike test_stops_when_cancel_races_in_during_the_judge_call
        (which pokes state.cancel_event.set() directly), this exercises the
        real agent.cancel() RPC path end-to-end.
        """
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(
            side_effect=[
                {"final_response": "did step 1", "messages": []},
                {"final_response": "did step 2, done", "messages": []},
                {"final_response": "did step 3, done", "messages": []},
            ]
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        loop = asyncio.get_running_loop()

        def fake_judge(goal, last_response, **kwargs):
            # Simulate a real client session/cancel landing while this
            # turn's judge call is in flight, via the actual cancel() RPC
            # (not a direct cancel_event.set()).
            future = asyncio.run_coroutine_threadsafe(
                agent.cancel(new_resp.session_id), loop
            )
            future.result(timeout=2)
            return ("continue", "keep going", False, None, False)

        with patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert state.cancel_event.is_set()
        assert resp.stop_reason == "cancelled"
        assert state.agent.run_conversation.call_count == 1

    @pytest.mark.asyncio
    async def test_judge_exception_pauses_goal_without_escaping_prompt(self, agent):
        """Regression: evaluate_after_turn (specifically judge_goal) raising must not
        crash prompt() -- the turn still ends normally, the goal auto-pauses so the
        user can /goal resume, and no continuation turn is attempted.
        """
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "did step 1", "messages": []})
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("hermes_cli.goals.judge_goal", side_effect=RuntimeError("judge exploded")):
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 1

        texts = _agent_message_texts(mock_conn)
        assert any(t.startswith("⏸ Goal paused — judge error:") for t in texts)

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"

    @pytest.mark.asyncio
    async def test_interrupted_turn_pauses_goal_with_cli_parity_notice(self, agent):
        """CLI parity (cli_loops_mixin._maybe_continue_goal_after_turn): an
        interrupted turn auto-pauses an active goal instead of judging partial
        output, using the same reason and notice wording the CLI uses.
        """
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        placeholder = INTERRUPT_WAITING_FOR_MODEL_PREFIX + "5.0s elapsed)."
        state.agent.run_conversation = MagicMock(
            return_value={
                "final_response": placeholder,
                "interrupted": True,
                "messages": [],
            }
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("hermes_cli.goals.judge_goal") as mock_judge:
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        mock_judge.assert_not_called()

        texts = _agent_message_texts(mock_conn)
        assert any(
            t == "⏸ Goal paused — turn was interrupted. "
            "Use /goal resume to continue, or /goal clear to stop."
            for t in texts
        )

        goal_state = load_goal(new_resp.session_id)
        assert goal_state.status == "paused"
        assert goal_state.paused_reason == "user-interrupted (Ctrl+C)"

    @pytest.mark.asyncio
    async def test_interrupt_placeholder_is_not_sent_to_the_goal_judge(self, agent):
        """Regression: an internal "waiting for model response" interrupt
        placeholder (not real assistant prose) must never reach the goal
        judge as if it were the model's genuine answer.
        """
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        placeholder = INTERRUPT_WAITING_FOR_MODEL_PREFIX + "5.0s elapsed)."
        state.agent.run_conversation = MagicMock(
            return_value={
                "final_response": placeholder,
                "interrupted": True,
                "messages": [],
            }
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("hermes_cli.goals.judge_goal") as mock_judge:
            prompt = [TextContentBlock(type="text", text="/goal Ship the widget")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        assert state.agent.run_conversation.call_count == 1
        mock_judge.assert_not_called()
