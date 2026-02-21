"""Tests for LLM Router core functionality."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from llm_router.config import load_profiles, load_providers, validate_config
from llm_router.router import AllProvidersExhausted, LLMRouter, RouterResult
from llm_router.tracker import StrikeTracker


# ── Config Tests ──────────────────────────────────────────────────────


class TestConfig:
    def test_load_default_providers(self):
        providers = load_providers()
        assert "gemini-free" in providers
        assert "openrouter-free" in providers
        assert "anthropic" in providers
        assert providers["gemini-free"]["sdk"] == "openai"
        assert providers["anthropic"]["sdk"] == "anthropic"

    def test_load_default_profiles(self):
        profiles = load_profiles()
        assert "free-only" in profiles
        assert "paid-first" in profiles
        assert "balanced" in profiles
        assert "gemini-free" in profiles["free-only"]["providers"]

    def test_custom_providers_merge(self, tmp_path):
        custom = {
            "providers": {
                "gemini-free": {"default_model": "gemini-2.5-flash"},
                "my-custom": {
                    "name": "Custom",
                    "env_key": "CUSTOM_KEY",
                    "sdk": "openai",
                    "default_model": "custom-model",
                },
            }
        }
        custom_file = tmp_path / "providers.yml"
        custom_file.write_text(yaml.dump(custom))

        providers = load_providers(custom_file)
        # Default overridden
        assert providers["gemini-free"]["default_model"] == "gemini-2.5-flash"
        # Custom added
        assert "my-custom" in providers
        # Others still present
        assert "anthropic" in providers

    def test_custom_profiles_merge(self, tmp_path):
        custom = {
            "profiles": {
                "my-profile": {
                    "description": "Custom profile",
                    "providers": ["anthropic"],
                }
            }
        }
        custom_file = tmp_path / "profiles.yml"
        custom_file.write_text(yaml.dump(custom))

        profiles = load_profiles(custom_file)
        assert "my-profile" in profiles
        assert "free-only" in profiles  # Defaults still present

    def test_validate_config_valid(self):
        providers = load_providers()
        profiles = load_profiles()
        warnings = validate_config(providers, profiles)
        assert len(warnings) == 0

    def test_validate_config_missing_provider(self):
        providers = {"foo": {"env_key": "FOO", "sdk": "openai"}}
        profiles = {"test": {"providers": ["foo", "missing"]}}
        warnings = validate_config(providers, profiles)
        assert any("missing" in w for w in warnings)


# ── Tracker Tests ─────────────────────────────────────────────────────


class TestTracker:
    def test_new_tracker_all_available(self):
        tracker = StrikeTracker()
        assert tracker.is_available("any-provider")

    def test_strike_makes_unavailable(self):
        tracker = StrikeTracker()
        tracker.strike("test-provider", "429 Too Many Requests")
        assert not tracker.is_available("test-provider")

    def test_expired_strike_auto_clears(self):
        tracker = StrikeTracker()
        # Manually set an expired strike
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        tracker.strikes["test-provider"] = {
            "struck_at": past.isoformat(),
            "reason": "test",
            "renews_at": past.isoformat(),
        }
        assert tracker.is_available("test-provider")
        assert "test-provider" not in tracker.strikes

    def test_persistence(self, tmp_path):
        state_file = tmp_path / "tracker.json"

        # Create and strike
        tracker1 = StrikeTracker(state_file)
        tracker1.strike("provider-a", "rate limited", retry_after_seconds=3600)

        # Load in new instance
        tracker2 = StrikeTracker(state_file)
        assert not tracker2.is_available("provider-a")

    def test_daily_renewal(self):
        tracker = StrikeTracker()
        tracker.strike("test", "quota exceeded", renewal_policy="daily")
        strike = tracker.strikes["test"]
        renews = datetime.fromisoformat(strike["renews_at"])
        now = datetime.now(timezone.utc)
        # Should renew at midnight UTC tomorrow
        assert renews.hour == 0
        assert renews.minute == 0
        assert renews.date() == (now.date() + timedelta(days=1))

    def test_get_status(self):
        tracker = StrikeTracker()
        tracker.strike("a", "test")
        status = tracker.get_status()
        assert status["active_strikes"] == 1
        assert "a" in status["strikes"]


# ── Router Tests ──────────────────────────────────────────────────────


class TestRouter:
    def test_init_loads_config(self):
        router = LLMRouter()
        assert len(router.providers) > 0
        assert len(router.profiles) > 0

    def test_unknown_profile_raises(self):
        router = LLMRouter()
        with pytest.raises(KeyError, match="nonexistent"):
            router.call("nonexistent", messages=[{"role": "user", "content": "hi"}])

    def test_no_api_keys_exhausts(self):
        """With no env vars set, all providers should be skipped."""
        router = LLMRouter()
        # Clear any real env vars
        env_keys = [p.get("env_key") for p in router.providers.values() if p.get("env_key")]
        with patch.dict(os.environ, {k: "" for k in env_keys if k}, clear=False):
            # Remove the keys entirely
            for k in env_keys:
                if k and k in os.environ:
                    del os.environ[k]
            # Need a fresh router since clients might be cached
            router2 = LLMRouter()
            with pytest.raises(AllProvidersExhausted):
                router2.call("free-only", messages=[{"role": "user", "content": "test"}])

    def test_list_available_no_keys(self):
        router = LLMRouter()
        env_keys = [p.get("env_key") for p in router.providers.values() if p.get("env_key")]
        clean_env = {k: "" for k in env_keys if k}
        with patch.dict(os.environ, clean_env, clear=False):
            for k in env_keys:
                if k and k in os.environ:
                    del os.environ[k]
            router2 = LLMRouter()
            available = router2.list_available("free-only")
            assert all(not p["is_available"] for p in available)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_skips_struck_provider(self, tmp_path):
        tracker_file = tmp_path / "tracker.json"

        router = LLMRouter(tracker_file=str(tracker_file))

        # Strike openrouter
        router.tracker.strike(
            "openrouter-free", "429", retry_after_seconds=3600
        )

        available = router.list_available("free-only")
        openrouter = next(p for p in available if p["id"] == "openrouter-free")
        assert not openrouter["is_available"]

    def test_router_result_dataclass(self):
        result = RouterResult(
            text="Hello!",
            provider_id="test",
            provider_name="Test Provider",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        assert result.text == "Hello!"
        assert result.provider_name == "Test Provider"

    def test_rate_limit_detection(self):
        router = LLMRouter()

        # 429 error
        err = Exception("Error code: 429 - Rate limit exceeded")
        is_rl, retry = router._is_rate_limit_error(err)
        assert is_rl

        # Quota error
        err = Exception("Quota exceeded for this model")
        is_rl, retry = router._is_rate_limit_error(err)
        assert is_rl

        # Normal error
        err = Exception("Connection timeout")
        is_rl, retry = router._is_rate_limit_error(err)
        assert not is_rl

    def test_rate_limit_with_retry_after(self):
        router = LLMRouter()
        err = Exception("429 Too Many Requests. Retry-After: 30")
        is_rl, retry = router._is_rate_limit_error(err)
        assert is_rl
        assert retry == 30
