"""``build_turn_context`` stubs prior-turn tool-result images.

"Pixels once, then stub" (docs/plans/2026-09-03-hermes-image-pixels-once.md,
switchboard repo): the turn-boundary call in agent/turn_context.py must
collapse any image left over from an earlier, completed turn before the
new user message is appended — and it must mutate the caller's own history
dicts in place, since `conversation_history` is passed straight through as
`state.history` by the ACP/CLI callers and only shallow-copied here.

Reuses the `_FakeAgent` / `_build` fixture harness from test_turn_context.py
rather than re-deriving the minimal agent stand-in.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.agent.test_turn_context import _build, _FakeAgent, _stub_runtime_main  # noqa: F401


def _prior_turn_image_tool_msg() -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {
                "type": "text",
                "text": (
                    "Image loaded into your context — you can see it "
                    "natively now.\n\nSource file: /tmp/prior-shot.png"
                ),
            },
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


class TestTurnContextStubsPriorTurnImages:
    def test_prior_turn_tool_image_stubbed_and_new_user_message_appended_after(self):
        agent = _FakeAgent()
        prior_tool_msg = _prior_turn_image_tool_msg()
        conversation_history = [
            {"role": "user", "content": "attach this screenshot"},
            {"role": "assistant", "content": "let me look", "tool_calls": []},
            prior_tool_msg,
            {"role": "assistant", "content": "it says hello"},
        ]

        ctx = _build(agent, conversation_history=conversation_history)

        # The stub landed on the tool message, ahead of the new user turn.
        stubbed = ctx.messages[2]
        assert stubbed["role"] == "tool"
        assert isinstance(stubbed["content"], str)
        assert "/tmp/prior-shot.png" in stubbed["content"]
        assert (
            "Call vision_analyze on this path if you need to look again."
            in stubbed["content"]
        )

        # The new user message was appended after the (now-stubbed) history.
        assert ctx.messages[-1]["role"] == "user"
        assert ctx.messages[-1]["content"] == "hello"
        assert ctx.current_turn_user_idx == len(ctx.messages) - 1

        # Mutated in place: the caller's own history dict reflects the stub
        # too, not just build_turn_context's local `messages` copy.
        assert conversation_history[2] is stubbed
        assert isinstance(conversation_history[2]["content"], str)

    def test_prior_turn_image_stub_persisted_through_agent_session_db(self):
        agent = _FakeAgent()
        agent._session_db = MagicMock()
        agent.session_id = "sess-persist"
        conversation_history = [
            {"role": "user", "content": "attach this screenshot"},
            {"role": "assistant", "content": "let me look", "tool_calls": []},
            _prior_turn_image_tool_msg(),
        ]

        ctx = _build(agent, conversation_history=conversation_history)

        stubbed = ctx.messages[2]
        agent._session_db.rewrite_message_content.assert_called_once_with(
            "sess-persist",
            "call_1",
            stubbed["content"],
            expected_content=_prior_turn_image_tool_msg()["content"],
        )

    def test_no_prior_image_leaves_history_untouched(self):
        agent = _FakeAgent()
        conversation_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        snapshot = [dict(m) for m in conversation_history]

        ctx = _build(agent, conversation_history=conversation_history)

        assert conversation_history == snapshot
        assert ctx.messages[-1]["content"] == "hello"
