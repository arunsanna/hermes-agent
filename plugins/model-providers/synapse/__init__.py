"""ArunLabs Synapse provider profile.

Synapse owns its live model catalogue and accepted reasoning levels. Hermes
discovers models from Synapse's chat-only endpoint and forwards the selected
``reasoning_effort`` as the top-level OpenAI-compatible request field.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

def _synapse_reasoning_effort(reasoning_config: dict | None) -> str | None:
    """Return the caller's selected level; Synapse validates its own schema."""
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return None
    effort = (reasoning_config.get("effort") or "").strip().lower()
    return effort or None


class SynapseProfile(ProviderProfile):
    """Synapse — live model discovery and top-level reasoning effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = _synapse_reasoning_effort(reasoning_config)
        if effort is None:
            return {}, {}
        return {}, {"reasoning_effort": effort}


synapse = SynapseProfile(
    name="synapse",
    aliases=("arunlabs", "forge"),
    env_vars=("SYNAPSE_API_KEY",),
    display_name="ArunLabs Synapse",
    description="Self-hosted OpenAI-compatible gateway (forge k3s)",
    base_url="https://synapse.arunlabs.com/v1",
    models_url="https://synapse.arunlabs.com/v1/chat/models",
    fallback_models=(),
    supports_vision=True,
)

register_provider(synapse)
