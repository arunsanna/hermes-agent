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


class TestSynapseCatalogIsTheSourceOfTruth:
    """Synapse advertises ``capabilities.reasoning_levels`` per model on its
    ``/v1/chat/models`` catalog and rejects anything else with HTTP 400
    (``reasoning_effort must be one of: minimal, low, medium, high``). Hermes'
    own ladder (``xhigh``/``max``/``ultra``, config defaults) must be clamped
    onto that list before the request leaves the process."""

    QWEN = "qwen3.8-abliterated"

    @pytest.fixture(autouse=True)
    def _catalog(self, synapse_profile, monkeypatch):
        import hermes_cli.models as models_module

        models_module._reset_synapse_catalog_cache_for_testing()
        self.catalog = {
            self.QWEN: {
                "supports_reasoning_levels": True,
                "reasoning_levels": ["minimal", "low", "medium", "high"],
            },
            "no-levels-model": {"supports_reasoning_levels": False},
        }
        self.fetches = 0

        def fake_fetch(*, base_url, api_key, timeout):
            self.fetches += 1
            return self.catalog

        monkeypatch.setattr(models_module, "_fetch_synapse_capabilities", fake_fetch)

    @pytest.mark.parametrize("effort", ["xhigh", "max", "ultra"])
    def test_levels_above_the_catalog_clamp_to_high(self, synapse_profile, effort):
        _, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model=self.QWEN
        )
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
    def test_advertised_levels_pass_through(self, synapse_profile, effort):
        _, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model=self.QWEN
        )
        assert top_level == {"reasoning_effort": effort}

    def test_model_without_levels_omits_the_field(self, synapse_profile):
        _, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="no-levels-model",
        )
        assert top_level == {}

    def test_unknown_catalog_passes_the_request_through(self, synapse_profile):
        self.catalog = None
        _, top_level = synapse_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}, model=self.QWEN
        )
        assert top_level == {"reasoning_effort": "max"}

    def test_catalog_is_fetched_once_per_process(self, synapse_profile):
        for _ in range(3):
            synapse_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": "max"}, model=self.QWEN
            )
        assert self.fetches == 1
