"""SessionDB.rewrite_message_content — durable persistence for image stubs.

"Pixels once, then stub" (docs/plans/2026-09-03-hermes-image-pixels-once.md,
switchboard repo): once agent.context_compressor.stub_prior_turn_tool_images
collapses a tool-result image to a text stub in memory, the DB row must be
rewritten too, or a later resume reloads the old base64 straight from
state.db. Keyed by (session_id, tool_call_id, role='tool') rather than the
row id — see the plan's S1 spike for why.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-stub", source="cli")
    yield d
    d.close()


def _image_content():
    return [
        {"type": "text", "text": "Image loaded.\n\nSource file: /tmp/shot.png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


class TestRewriteMessageContent:
    def test_rewrite_replaces_content_and_clears_api_content(self, db):
        db.append_message(
            "sess-stub",
            role="tool",
            content=_image_content(),
            tool_call_id="call_1",
            api_content="stale byte-identical replay",
        )

        stub_text = (
            "[image no longer in context] Image loaded. Source file: "
            "/tmp/shot.png\n\nCall vision_analyze on this path if you need "
            "to look again."
        )
        ok = db.rewrite_message_content("sess-stub", "call_1", stub_text)
        assert ok is True

        conversation = db.get_messages_as_conversation("sess-stub")
        tool_msgs = [m for m in conversation if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == stub_text
        assert "data:image" not in tool_msgs[0]["content"]
        assert "api_content" not in tool_msgs[0]

    def test_rewrite_unknown_tool_call_id_is_a_noop(self, db):
        db.append_message(
            "sess-stub", role="tool", content=_image_content(), tool_call_id="call_1",
        )
        ok = db.rewrite_message_content("sess-stub", "no-such-call", "stub")
        assert ok is False
        conversation = db.get_messages_as_conversation("sess-stub")
        tool_msgs = [m for m in conversation if m["role"] == "tool"]
        assert tool_msgs[0]["content"] == _image_content()
