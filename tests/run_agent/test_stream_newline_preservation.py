"""Newline-only stream deltas must survive ACP provisional buffering.

Regression guard for the collapsed-markdown-table bug (Switchboard, 2026-07-31):
vLLM (Synapse Qwen) streams newlines as standalone deltas ("\n", "\n\n").
While `_acp_provisional_stream_active` buffers visible text, deltas bypass
`_record_streamed_assistant_text`, so `_current_streamed_assistant_text`
never grows and the "first delta only" leading-newline lstrip in
`_fire_stream_delta` applied to EVERY delta — annihilating every
standalone newline. Tables rendered as one run-on line on every client.

Providers that glue newlines to text chunks (z.ai GLM) masked the bug.
"""

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._interrupt_requested = False
    return agent


# vLLM/Qwen delta shape captured live from synapse.arunlabs.com (2026-07-31):
# newlines arrive as standalone deltas, never glued to text.
QWEN_TABLE_DELTAS = [
    "|", " Fruit", " |", " Count", " |",
    "\n",
    "|", "---", "|", "---", "|",
    "\n",
    "|", " apple", " |", " ", "1", " |",
    "\n\n",
    "Total", ":", " ", "1", " fruit", ".",
]
EXPECTED_TEXT = "| Fruit | Count |\n|---|---|\n| apple | 1 |\n\nTotal: 1 fruit."


class TestProvisionalStreamNewlinePreservation:
    def test_standalone_newline_deltas_survive_provisional_buffering(self):
        agent = _make_agent()
        agent._acp_provisional_stream_active = True
        agent._acp_provisional_stream_buffer = []

        for delta in QWEN_TABLE_DELTAS:
            agent._fire_stream_delta(delta)

        buffered = "".join(
            payload
            for kind, payload in agent._acp_provisional_stream_buffer
            if kind == "stream_delta"
        )
        assert buffered == EXPECTED_TEXT, (
            f"Provisional buffer corrupted the stream: {buffered!r}"
        )

    def test_leading_newlines_still_stripped_at_stream_start(self):
        agent = _make_agent()
        agent._acp_provisional_stream_active = True
        agent._acp_provisional_stream_buffer = []

        # Model opens with bare newlines before the first visible text.
        for delta in ["\n\n", "\n", "Hello", "\n", "world"]:
            agent._fire_stream_delta(delta)

        buffered = "".join(
            payload
            for kind, payload in agent._acp_provisional_stream_buffer
            if kind == "stream_delta"
        )
        assert buffered == "Hello\nworld", (
            f"Leading-newline strip regressed: {buffered!r}"
        )

    def test_delivered_path_unaffected_when_not_provisional(self):
        agent = _make_agent()
        agent._acp_provisional_stream_active = False

        delivered: list = []
        agent.stream_delta_callback = delivered.append

        for delta in QWEN_TABLE_DELTAS:
            agent._fire_stream_delta(delta)

        assert "".join(delivered) == EXPECTED_TEXT, (
            f"Direct delivery corrupted the stream: {''.join(delivered)!r}"
        )

    def test_discarded_provisional_attempt_rearms_leading_strip(self):
        # A final-response candidate racing a required delegation is
        # discarded; the retried response is the stream start again, so its
        # leading newlines must be stripped exactly as pre-fix.
        agent = _make_agent()
        agent._acp_provisional_stream_active = True
        agent._acp_provisional_stream_buffer = []

        for delta in ["\n\n", "Stale", " candidate"]:
            agent._fire_stream_delta(delta)
        agent._finish_acp_provisional_stream(discard=True)

        assert agent._stream_visible_text_started is False, (
            "Discard must re-arm the stream-start leading-newline strip"
        )

        agent._acp_provisional_stream_active = True
        agent._acp_provisional_stream_buffer = []
        for delta in ["\n\n", "\n", "Fresh", "\n", "answer"]:
            agent._fire_stream_delta(delta)

        buffered = "".join(
            payload
            for kind, payload in agent._acp_provisional_stream_buffer
            if kind == "stream_delta"
        )
        assert buffered == "Fresh\nanswer", (
            f"Retried stream start not re-stripped: {buffered!r}"
        )

    def test_discard_after_delivered_text_keeps_flag(self):
        # Mid-turn: visible text was already delivered, then a later
        # provisional attempt is discarded. The flag must mirror the
        # accumulator (True) so mid-turn newlines keep flowing.
        agent = _make_agent()
        agent._acp_provisional_stream_active = False
        agent.stream_delta_callback = lambda _t: None
        agent._fire_stream_delta("Earlier segment.")
        assert agent._current_streamed_assistant_text

        agent._acp_provisional_stream_active = True
        agent._acp_provisional_stream_buffer = []
        agent._fire_stream_delta("stale")
        agent._finish_acp_provisional_stream(discard=True)

        assert agent._stream_visible_text_started is True, (
            "Discard must not re-arm the strip when text was already delivered"
        )
