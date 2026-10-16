"""Tests for named user-defined provider entries in the ACP model selector.

Named endpoints from the ``providers:`` mapping (and legacy
``custom_providers:`` list) are invisible to canonical provider enumeration,
so ``_build_model_state`` must append them explicitly for ACP clients to
offer them — the TUI ``/model`` picker already renders these entries
(#47039 implemented named endpoints for the TUI surface only).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.model_catalog import _named_custom_provider_catalogs
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from acp.schema import SessionModelState


MANTLE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


class TestNamedCustomProviderCatalogs:

    def test_cached_discovery_extends_declared_models(self, monkeypatch):
        """HAF-004: the named path is cache-only — a warmed shared-cache entry extends
        declared models with no live fetch on the session path."""
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "model-a",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.cached_fetch_api_models",
            return_value=["model-a", "model-b"],
        ) as cached, patch(
            "hermes_cli.model_switch_providers._fetch_picker_live_models",
            side_effect=AssertionError("HAF-004: no live fetch on the session path"),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert len(catalogs) == 1
        slug, label, models = catalogs[0]
        assert slug == "custom:relay"
        assert [m for m, _ in models] == ["model-a", "model-b"]
        cached.assert_called_once()
        assert cached.call_args.kwargs.get("cache_only") is True


    def test_disabled_provider_skipped(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "off": {
                    "name": "Disabled Endpoint",
                    "base_url": "https://off.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "m",
                    "enabled": False,
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.cached_fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_no_credential_and_no_declared_models_skipped(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = _cfg(
            providers={
                "bare": {
                    "name": "Bare",
                    "base_url": "https://bare.example/v1",
                    "key_env": "MISSING_KEY",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.cached_fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_legacy_custom_providers_list_included(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            custom_providers=[
                {
                    "name": "Legacy Endpoint",
                    "base_url": "https://legacy.example/v1",
                    "key_env": "SOME_KEY",
                    "model": "legacy-model",
                }
            ]
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.cached_fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            ("custom:legacy-endpoint", "Legacy Endpoint", [("legacy-model", "")])
        ]
    def test_no_key_ollama_provider_serves_cached_native_catalog(self):
        """HAF-004: a warmed native Ollama cache entry is served with no live probe."""
        cfg = _cfg(
            providers={
                "custom:ollama": {
                    "name": "Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models_local.should_use_ollama_native_catalog",
            return_value=True,
        ), patch(
            "hermes_cli.models.cached_fetch_api_models",
            return_value=["qwen3:1.7b"],
        ) as cached, patch(
            "hermes_cli.model_switch_providers._fetch_picker_live_models",
            side_effect=AssertionError("HAF-004: no live fetch on the session path"),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["qwen3:1.7b"]
        assert cached.call_args.kwargs.get("cache_only") is True
    def test_legacy_credentialless_ollama_serves_cached_native_catalog(self):
        cfg = _cfg(
            custom_providers=[
                {
                    "name": "Local Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            ]
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models_local.should_use_ollama_native_catalog",
            return_value=True,
        ), patch(
            "hermes_cli.models.cached_fetch_api_models",
            return_value=["qwen3:1.7b"],
        ) as cached, patch(
            "hermes_cli.model_switch_providers._fetch_picker_live_models",
            side_effect=AssertionError("HAF-004: no live fetch on the session path"),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["qwen3:1.7b"]
        cached.assert_called_once()
        assert cached.call_args.kwargs.get("cache_only") is True

    def test_native_empty_catalog_is_authoritative_over_default_model(self):
        cfg = _cfg(
            providers={
                "custom:ollama": {
                    "name": "Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "default_model": "saved:model",
                }
            }
        )
        from hermes_cli.model_switch_providers import _NativePickerModelList

        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models_local.should_use_ollama_native_catalog",
            return_value=True,
        ), patch(
            "hermes_cli.models.cached_fetch_api_models",
            return_value=_NativePickerModelList(),
        ):
            assert _named_custom_provider_catalogs() == [
                ("custom:ollama", "Ollama", [])
            ]


class TestModelStateIncludesNamedProviders:
    @pytest.mark.asyncio
    async def test_authoritative_empty_named_catalog_does_not_resurrect_current_model(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="saved:model", provider="ollama"
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:ollama", "Ollama", [])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == ""
        assert all(
            not item.model_id.startswith(("ollama:", "custom:ollama:"))
            for item in resp.models.available_models
        )

    @pytest.mark.asyncio
    async def test_named_provider_models_appear_in_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4", provider="openai-codex"
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[
                (
                    "custom:bedrock-mantle",
                    "AWS Bedrock Mantle",
                    [("openai.gpt-5.5", "")],
                )
            ],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        ids = [m.model_id for m in resp.models.available_models]
        # Current provider's models come first, named endpoints after.
        assert ids[0] == "openai-codex:gpt-5.4"
        assert "custom:bedrock-mantle:openai.gpt-5.5" in ids
        named = next(
            m
            for m in resp.models.available_models
            if m.model_id == "custom:bedrock-mantle:openai.gpt-5.5"
        )
        assert "AWS Bedrock Mantle" in (named.description or "")

    @pytest.mark.asyncio
    async def test_configured_provider_inventory_row_uses_custom_choice_id(self):
        """A ``providers:`` row must not expose its raw config key to ACP."""
        from hermes_cli.models import parse_model_input

        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="model-a", provider="relay")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        cfg = {
            "providers": {
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                }
            }
        }
        inventory = {
            "providers": [{"slug": "relay", "name": "Relay", "is_user_defined": True, "models": ["model-a"]}]
        }

        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.inventory.build_models_payload", return_value=inventory
        ), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:relay", "Relay", [("model-a", "")])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")
            choice_ids = [item.model_id for item in resp.models.available_models]
            provider, model = parse_model_input(resp.models.current_model_id, "relay")

        assert choice_ids == ["custom:relay:model-a"]
        assert provider == "custom:relay"
        assert model == "model-a"

    def test_selector_choice_id_round_trips_through_parse_model_input(self):
        """The encoded choice id must resolve back to the named provider."""
        from hermes_cli.models import parse_model_input

        choice_id = "custom:bedrock-mantle:openai.gpt-5.5"
        cfg = {
            "providers": {
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": "https://bedrock.example/v1",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            provider, model = parse_model_input(choice_id, "bedrock")
        assert provider == "custom:bedrock-mantle"
        assert model == "openai.gpt-5.5"

    def test_selector_choice_id_round_trips_colon_bearing_custom_identity(self):
        """Configured provider and model IDs may both contain colons."""
        from hermes_cli.models import parse_model_input

        cfg = {
            "providers": {
                "local-127.0.0.1:11434": {
                    "name": "Local Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            provider, model = parse_model_input(
                "custom:local-127.0.0.1:11434:qwen3:1.7b", "custom"
            )
        assert provider == "custom:local-127.0.0.1:11434"
        assert model == "qwen3:1.7b"

    @pytest.mark.asyncio
    async def test_named_entry_shadowing_a_canonical_provider_keeps_the_canonical_session(self):
        """``providers.openrouter:`` (a proxy) must not swallow the canonical OpenRouter rows nor
        relabel a session running on openrouter.ai as ``custom:openrouter`` (that id resolves to
        the proxy base_url)."""
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="model-c", provider="openrouter", base_url="https://openrouter.ai/api/v1")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        inventory = {"providers": [
            {"slug": "openrouter", "name": "OpenRouter", "is_user_defined": False, "models": ["model-c"]},
            {"slug": "custom:openrouter", "name": "openrouter", "is_user_defined": True,
             "api_url": "https://or.example/api/v1", "models": ["model-a"]},
        ]}

        with patch("hermes_cli.inventory.build_models_payload", return_value=inventory), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:openrouter", "openrouter", [("model-a", "")])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert resp.models.current_model_id == "openrouter:model-c"
        assert [m.model_id for m in resp.models.available_models] == ["openrouter:model-c", "custom:openrouter:model-a"]


class TestCacheOnlyFirstTurnContract:
    """HAF-004 hit/miss/stale/error matrix for the session/new+load model state.

    The first usable turn never waits on unselected-provider discovery: a cache hit
    serves warmed rows, a miss serves curated/declared rows, and any error falls back
    to the exact current-model row (id round-trips, no re-routing)."""

    def test_canonical_payload_is_cache_only(self):
        """The canonical inventory read must agree with the named cache-only path:
        ``non_blocking_catalogs=True`` with no live probes."""
        from acp_adapter.model_catalog import build_model_state

        seen = {}

        def _recording_payload(ctx, **kwargs):
            seen.update(kwargs)
            return {"providers": []}

        with patch("hermes_cli.inventory.build_models_payload", side_effect=_recording_payload), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs", return_value=[]
        ):
            state = build_model_state("test-model-1", "openrouter", "")

        assert seen.get("non_blocking_catalogs") is True
        assert seen.get("refresh") is False
        assert seen.get("probe_custom_providers") is False
        assert seen.get("probe_current_custom_provider") is False
        # Miss with an empty inventory: the exact current model is still the row.
        assert state is not None
        assert state.current_model_id == "openrouter:test-model-1"
        assert [m.model_id for m in state.available_models] == ["openrouter:test-model-1"]

    def test_inventory_error_returns_none(self):
        """Error: any inventory failure maps to ``None`` (caller emits current-only)."""
        from acp_adapter.model_catalog import build_model_state

        with patch(
            "hermes_cli.inventory.build_models_payload", side_effect=RuntimeError("disk gone")
        ):
            assert build_model_state("test-model-1", "openrouter", "") is None

    @pytest.mark.asyncio
    async def test_server_error_falls_back_to_current_model_only(self):
        """Error at the session layer: exact current id, round-trips, no re-routing."""
        from hermes_cli.models import parse_model_input

        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="gpt-5.4", provider="openai-codex")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        with patch(
            "acp_adapter.model_catalog.build_model_state", side_effect=RuntimeError("boom")
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [m.model_id for m in resp.models.available_models] == ["openai-codex:gpt-5.4"]
        provider, model = parse_model_input(resp.models.current_model_id, "")
        assert (provider, model) == ("openai-codex", "gpt-5.4")

    @pytest.mark.asyncio
    async def test_miss_keeps_current_model_without_declared_or_cached_rows(self):
        """Miss: an unselected endpoint with no declared models and a cold cache is
        dropped, while the exact current model id is preserved."""
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="test-model-1", provider="openrouter")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        cfg = _cfg(
            providers={
                "cold": {
                    "name": "Cold Endpoint",
                    "base_url": "https://cold-cache.example/v1",
                    "api_key": "dummy",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.cached_fetch_api_models", return_value=None
        ), patch(
            "hermes_cli.model_switch_providers._fetch_picker_live_models",
            side_effect=AssertionError("HAF-004: no live fetch on the session path"),
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert resp.models.current_model_id == "openrouter:test-model-1"
        assert all(not m.model_id.startswith("custom:cold:") for m in resp.models.available_models)

    def test_stale_cache_served_and_revalidated_in_background(self, monkeypatch, tmp_path):
        """Stale: a past-TTL same-credential entry is served now (no network) and a
        background revalidation is scheduled for the next open."""
        import time

        import hermes_cli.models as models_mod
        from hermes_cli.models import (
            _cache_entry,
            _custom_endpoint_fingerprint,
            _store_cache_entry,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("SOME_KEY", "k")
        base_url = "https://stale-cache.example/v1"
        fp = _custom_endpoint_fingerprint("k", None, None)
        aged_at = time.time() - (models_mod._PROVIDER_MODELS_CACHE_TTL + 60)
        _store_cache_entry(f"custom:{base_url}#{fp}", _cache_entry(fp, ["warmed-a"], aged_at))

        cfg = _cfg(
            providers={
                "staleep": {
                    "name": "Stale Endpoint",
                    "base_url": base_url,
                    "key_env": "SOME_KEY",
                    "default_model": "declared-m",
                }
            }
        )
        spawns = []
        with patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
            models_mod, "_spawn_swr_refresh", side_effect=lambda *a, **k: spawns.append((a, k))
        ), patch(
            "hermes_cli.model_switch_providers._fetch_picker_live_models",
            side_effect=AssertionError("HAF-004: no live fetch on the session path"),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [(s, lbl, [m for m, _ in ms]) for s, lbl, ms in catalogs] == [
            ("custom:staleep", "Stale Endpoint", ["declared-m", "warmed-a"])
        ]
        assert spawns, "stale entry must schedule a background revalidation"
