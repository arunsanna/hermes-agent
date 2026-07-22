"""Unit tests for the Z.AI (GLM) provider profile's reasoning wiring.

Z.AI accepts ``max|xhigh|high|medium|low|minimal|none`` for a top-level
``reasoning_effort``, but only "high" and "max" are distinct on the wire:
low/medium map to high server-side, xhigh maps to max, and ultra is our own
alias for max, and none/minimal skip thinking. The profile clamps to the
meaningful values and SENDS ``reasoning_effort: "none"`` for none/minimal or
disabled thinking — omission would fall back to the server default (max).

Regression context: this logic previously lived as a dead ``provider_name
== "zai"`` branch in ``ChatCompletionsTransport.build_kwargs()``'s legacy
fallback path. Because "zai" is a registered ``ProviderProfile``, the real
call site (``chat_completion_helpers.py``) always takes the profile path and
never reaches the legacy branch — so that code never ran for real traffic.
The fix lives on the profile's ``build_api_kwargs_extras()`` instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def zai_profile():
    """Resolve the registered Z.AI profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Z.AI profile. Going through ``get_provider_profile`` keeps the test
    honest: if the registered class is ever swapped for a plain
    ``ProviderProfile`` the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("zai")
    assert profile is not None, "zai provider profile must be registered"
    return profile


class TestZaiReasoningEffortMapping:
    """``build_api_kwargs_extras`` clamps reasoning_config.effort correctly."""

    def test_default_no_config_is_max(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(reasoning_config=None)
        assert top_level == {"reasoning_effort": "max"}
        assert extra_body == {}

    def test_enabled_without_effort_defaults_to_max(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}
        )
        assert top_level == {"reasoning_effort": "max"}
        assert extra_body == {}

    def test_high_kept_as_is(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert top_level == {"reasoning_effort": "high"}
        assert extra_body == {}

    def test_max_kept_as_is(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}
        )
        assert top_level == {"reasoning_effort": "max"}
        assert extra_body == {}

    @pytest.mark.parametrize("effort", ["xhigh", "ultra"])
    def test_xhigh_and_ultra_clamp_to_max(self, zai_profile, effort):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top_level == {"reasoning_effort": "max"}
        assert extra_body == {}

    @pytest.mark.parametrize("effort", ["low", "medium"])
    def test_low_and_medium_clamp_to_high(self, zai_profile, effort):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top_level == {"reasoning_effort": "high"}
        assert extra_body == {}

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_none_and_minimal_send_none(self, zai_profile, effort):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top_level == {"reasoning_effort": "none"}
        assert extra_body == {}

    def test_unrecognized_effort_falls_back_to_max(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"}
        )
        assert top_level == {"reasoning_effort": "max"}
        assert extra_body == {}

    def test_disabled_sends_none(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert top_level == {"reasoning_effort": "none"}
        assert extra_body == {}

    def test_disabled_ignores_effort(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        )
        assert top_level == {"reasoning_effort": "none"}
        assert extra_body == {}


class TestZaiFullKwargsIntegration:
    """The transport's full kwargs carry reasoning_effort through the real
    profile-dispatch path (``build_kwargs(..., provider_profile=zai_profile)``),
    mirroring how ``chat_completion_helpers.py`` invokes it in production."""

    def _build(self, zai_profile, reasoning_config):
        from agent.transports.chat_completions import ChatCompletionsTransport

        return ChatCompletionsTransport().build_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config=reasoning_config,
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )

    def test_default_lands_max_in_api_kwargs(self, zai_profile):
        kwargs = self._build(zai_profile, None)
        assert kwargs["reasoning_effort"] == "max"

    def test_high_lands_in_api_kwargs(self, zai_profile):
        kwargs = self._build(zai_profile, {"enabled": True, "effort": "high"})
        assert kwargs["reasoning_effort"] == "high"

    def test_xhigh_lands_as_max_in_api_kwargs(self, zai_profile):
        kwargs = self._build(zai_profile, {"enabled": True, "effort": "xhigh"})
        assert kwargs["reasoning_effort"] == "max"

    def test_low_lands_as_high_in_api_kwargs(self, zai_profile):
        kwargs = self._build(zai_profile, {"enabled": True, "effort": "low"})
        assert kwargs["reasoning_effort"] == "high"

    def test_disabled_sends_none_in_api_kwargs(self, zai_profile):
        kwargs = self._build(zai_profile, {"enabled": False})
        assert kwargs["reasoning_effort"] == "none"


class TestNonZaiProfileUnaffected:
    """A profile without a build_api_kwargs_extras override (e.g. a plain
    ProviderProfile for an unregistered/custom provider) must not emit
    reasoning_effort from the same reasoning_config."""

    def test_plain_profile_emits_nothing(self):
        from providers.base import ProviderProfile

        plain = ProviderProfile(name="some-other-provider")
        extra_body, top_level = plain.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}
        )
        assert extra_body == {}
        assert top_level == {}
