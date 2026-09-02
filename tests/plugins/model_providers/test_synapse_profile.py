"""Wire-contract tests for the ArunLabs Synapse provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def synapse_profile():
    """Resolve Synapse through Hermes' real provider discovery path."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("synapse")
    assert profile is not None, "synapse provider profile must be registered"
    return profile


class TestSynapseReasoningWireShape:
    def test_model_discovery_uses_the_live_chat_catalog(self, synapse_profile):
        assert synapse_profile.models_url == (
            "https://synapse.arunlabs.com/v1/chat/models"
        )
        assert synapse_profile.fallback_models == ()

    def test_no_preference_omits_reasoning_effort(self, synapse_profile):
        extra_body, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config=None
        )
        assert extra_body == {}
        assert top_level == {}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "future-level"]
    )
    def test_selected_effort_is_sent_at_top_level(self, synapse_profile, effort):
        extra_body, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": effort}

    def test_disabled_reasoning_omits_reasoning_effort(self, synapse_profile):
        extra_body, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        )
        assert extra_body == {}
        assert top_level == {}
