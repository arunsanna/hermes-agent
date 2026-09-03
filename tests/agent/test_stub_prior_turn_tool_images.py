"""Stub prior-turn tool-result images at the turn boundary.

"Pixels once, then stub" (docs/plans/2026-09-03-hermes-image-pixels-once.md,
switchboard repo): a turn keeps full pixels for its own tool iterations, but
by the NEXT turn boundary every earlier tool-result image collapses to a
text stub carrying the source path, so later requests stop re-sending the
data URL. The model can still re-load the file via vision_analyze.
"""

from __future__ import annotations

from agent.context_compressor import stub_prior_turn_tool_images


def _native_vision_tool_msg(path: str = "/tmp/screenshot.png") -> dict:
    # Shape produced by _build_native_vision_tool_result + the unwrap in
    # agent/tool_executor.py: a plain OpenAI-style content-parts list, text
    # part carrying the "Source file: <path>" line added in Task 1.1.
    return {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {
                "type": "text",
                "text": (
                    "Image loaded into your context — you can see it "
                    "natively now. Use your built-in vision to answer "
                    "the user.\n\nQuestion: what does this say?"
                    f"\n\nSource file: {path}"
                ),
            },
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


def _multimodal_dict_tool_msg(path: str = "/tmp/screenshot.png") -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Image attached natively for the main model."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "text_summary": "Image attached natively for the main model (1.2 KB).",
            "meta": {"image_url": path, "size_bytes": 1234, "native_vision": True},
        },
    }


def _screenshot_path_meta_tool_msg(path: str = "/tmp/shot123.png") -> dict:
    # Shape produced by tools/computer_use/tool.py's _capture_response and
    # tools/browser_use_cli.py's _native_screenshot_result: the identical
    # `_multimodal` envelope, but meta carries the on-disk path under
    # `screenshot_path`, not `image_url`.
    return {
        "role": "tool",
        "tool_call_id": "call_5",
        "content": {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "capture mode=vision 1280x800"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "text_summary": "capture mode=vision 1280x800",
            "meta": {"screenshot_path": path, "native_vision": True},
        },
    }


class TestStubPriorTurnToolImages:
    def test_list_content_image_becomes_single_string_stub(self):
        messages = [_native_vision_tool_msg()]
        count = stub_prior_turn_tool_images(messages)
        assert count == 1
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "what does this say?" in content
        assert "/tmp/screenshot.png" in content
        assert "Call vision_analyze on this path if you need to look again." in content

    def test_multimodal_dict_content_becomes_single_string_stub(self):
        messages = [_multimodal_dict_tool_msg()]
        count = stub_prior_turn_tool_images(messages)
        assert count == 1
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "Image attached natively for the main model" in content
        assert "/tmp/screenshot.png" in content
        assert "Call vision_analyze on this path if you need to look again." in content

    def test_multimodal_dict_with_screenshot_path_meta_keeps_reload_hint(self):
        # computer_use / browser_use_cli screenshots key their on-disk path
        # as meta.screenshot_path rather than meta.image_url — the stub
        # must still recover it and keep the reload hint.
        messages = [_screenshot_path_meta_tool_msg()]
        count = stub_prior_turn_tool_images(messages)
        assert count == 1
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "/tmp/shot123.png" in content
        assert "Call vision_analyze on this path if you need to look again." in content

    def test_tool_message_without_image_untouched(self):
        msg = {"role": "tool", "tool_call_id": "call_3", "content": "plain text result"}
        messages = [msg]
        count = stub_prior_turn_tool_images(messages)
        assert count == 0
        assert messages[0] is msg
        assert messages[0]["content"] == "plain text result"

    def test_tool_message_list_content_without_image_untouched(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_4",
            "content": [{"type": "text", "text": "no image here"}],
        }
        messages = [msg]
        count = stub_prior_turn_tool_images(messages)
        assert count == 0
        assert messages[0]["content"] == [{"type": "text", "text": "no image here"}]

    def test_user_message_with_image_untouched(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
        messages = [msg]
        count = stub_prior_turn_tool_images(messages)
        assert count == 0
        assert messages[0]["content"] == msg["content"]

    def test_api_content_sidecar_dropped_on_stub(self):
        msg = _native_vision_tool_msg()
        msg["api_content"] = "stale byte-identical replay content"
        messages = [msg]
        stub_prior_turn_tool_images(messages)
        assert "api_content" not in messages[0]

    def test_idempotent_second_pass_changes_nothing(self):
        messages = [_native_vision_tool_msg(), _multimodal_dict_tool_msg()]
        first = stub_prior_turn_tool_images(messages)
        assert first == 2
        snapshot = [dict(m) for m in messages]
        second = stub_prior_turn_tool_images(messages)
        assert second == 0
        assert messages == snapshot
