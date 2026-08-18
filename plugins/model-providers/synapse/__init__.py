"""ArunLabs Synapse provider profile.

Synapse is a self-hosted OpenAI-compatible gateway (llama-router on forge
k3s) serving Qwen3.8 and friends. Without this profile it resolved to the
base profile, which emits no reasoning fields at all, so a user's effort
selection was accepted by the UI, carried all the way to the ACP adapter in
``HERMES_SESSION_REASONING_EFFORT`` (read at ``acp_adapter/session.py``), and
then silently dropped before the HTTP request — the picker advertised a
control that changed nothing.

Synapse takes the knob as a **chat-template** argument, not as OpenAI's
top-level ``reasoning_effort`` and not as ``extra_body.reasoning``:

    {"chat_template_kwargs": {"reasoning_effort": "low" | "medium" | "xhigh"}}

``chat_template_kwargs`` has no named parameter in the OpenAI SDK, so it is
returned in the ``extra_body`` half of the tuple; the transport merges that
dict into the top level of the outgoing JSON body, which is where the server
reads it.

Measured against ``synapse.arunlabs.com`` on 2026-08-15 with
``qwen3.8-uncensored`` (one sample per level, same prompt): ``low`` produced
147 completion tokens / 224 reasoning characters, ``xhigh`` produced 211 /
381. The field is honored, which supersedes the 2026-08-09 finding that
top-level ``reasoning_effort`` behaved as a binary switch on this endpoint.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# The exact values the server's chat template accepts. Anything outside this
# set is not sent — a rejected or ignored value would be indistinguishable
# from the silent drop this profile exists to fix.
_SYNAPSE_EFFORTS = ("low", "medium", "xhigh")


def _synapse_reasoning_effort(reasoning_config: dict | None) -> str | None:
    """Collapse Hermes' effort scale onto Synapse's three accepted levels.

    Hermes' ladder is none < minimal < low < medium < high < xhigh < max <
    ultra; Synapse offers only low/medium/xhigh. The upper band clamps to
    ``xhigh`` (mirroring how the Z.AI profile clamps to GLM-5.2's top tier).

    Synapse exposes no "thinking off" switch through this knob, so an
    explicitly disabled or ``none`` preference maps to ``low`` — the least
    thinking the server can actually be asked for. Omitting the field instead
    would fall back to the server default, which is *more* thinking than the
    user asked for, i.e. further from their intent.
    """
    if not isinstance(reasoning_config, dict):
        return None

    if reasoning_config.get("enabled") is False:
        return "low"

    effort = (reasoning_config.get("effort") or "").strip().lower()
    if not effort:
        return None
    if effort in _SYNAPSE_EFFORTS:
        return effort
    if effort in {"none", "minimal"}:
        return "low"
    if effort in {"high", "max", "ultra"}:
        return "xhigh"
    return None


class SynapseProfile(ProviderProfile):
    """Synapse — reasoning effort via chat_template_kwargs."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = _synapse_reasoning_effort(reasoning_config)
        if effort is None:
            return {}, {}
        return {"chat_template_kwargs": {"reasoning_effort": effort}}, {}


synapse = SynapseProfile(
    name="synapse",
    aliases=("arunlabs", "forge"),
    env_vars=("SYNAPSE_API_KEY",),
    display_name="ArunLabs Synapse",
    description="Self-hosted OpenAI-compatible gateway (forge k3s)",
    base_url="https://synapse.arunlabs.com/v1",
    # The served catalogue is discovered live from /v1/chat/models; this is
    # only the offline fallback for the currently loaded default.
    fallback_models=("qwen3.8-uncensored",),
    # qwen3.8-uncensored advertises capabilities.supports_vision = true.
    supports_vision=True,
)

register_provider(synapse)
