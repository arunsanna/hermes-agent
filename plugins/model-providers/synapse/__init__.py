"""ArunLabs Synapse provider profile.

Synapse owns its live model catalogue and accepted reasoning levels: each
``/v1/chat/models`` entry advertises ``capabilities.reasoning_levels`` and the
gateway answers HTTP 400 to anything else. The selected ``reasoning_effort`` is
clamped onto that catalogue by :func:`hermes_cli.models.synapse_wire_reasoning_effort`
(shared with HERMES_HOME plugin overrides) and sent as the top-level
OpenAI-compatible request field.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class SynapseProfile(ProviderProfile):
    """Synapse — live model discovery and catalogue-clamped reasoning effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from hermes_cli.models import synapse_wire_reasoning_effort

        wire = synapse_wire_reasoning_effort(
            reasoning_config, model=model, base_url=base_url or self.base_url
        )
        if wire is None:
            return {}, {}
        return {}, {"reasoning_effort": wire}


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
