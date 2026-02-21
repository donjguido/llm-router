"""LLM Router — Priority-based API routing with automatic fallback.

Tries providers in profile order, handles rate limits, and abstracts
away differences between OpenAI-compatible and Anthropic SDKs.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import load_profiles, load_providers, validate_config
from .tracker import StrikeTracker


@dataclass
class RouterResult:
    """Result from a successful LLM API call."""

    text: str
    provider_id: str
    provider_name: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class AllProvidersExhausted(Exception):
    """Raised when no provider could fulfill the request."""

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        providers = [a["provider"] for a in attempts]
        super().__init__(
            f"All providers exhausted. Tried: {', '.join(providers)}. "
            f"Details: {attempts}"
        )


class LLMRouter:
    """Route LLM requests across multiple providers with automatic fallback.

    Usage:
        router = LLMRouter()
        result = router.call("free-only", messages=[
            {"role": "user", "content": "Hello!"}
        ], temperature=0.7, max_tokens=1000)
        print(result.text)
        print(f"Served by: {result.provider_name} ({result.model})")
    """

    def __init__(
        self,
        providers_file: str | None = None,
        profiles_file: str | None = None,
        tracker_file: str | None = None,
    ):
        """Initialize the router.

        Args:
            providers_file: Path to custom providers.yml (merged with defaults).
            profiles_file: Path to custom profiles.yml (merged with defaults).
            tracker_file: Path to tracker-state.json for rate limit persistence.
        """
        self.providers = load_providers(providers_file)
        self.profiles = load_profiles(profiles_file)
        self.tracker = StrikeTracker(tracker_file)

        # Validate config
        warnings = validate_config(self.providers, self.profiles)
        for w in warnings:
            print(f"[llm-router] WARNING: {w}", file=sys.stderr)

        # Cache initialized clients
        self._clients: dict[str, Any] = {}

    def _get_api_key(self, provider_id: str) -> str | None:
        """Get API key from environment for a provider."""
        env_key = self.providers[provider_id].get("env_key")
        if not env_key:
            return None
        return os.environ.get(env_key)

    def _get_client(self, provider_id: str):
        """Get or create an API client for a provider.

        Returns (client, is_anthropic) tuple.
        """
        if provider_id in self._clients:
            return self._clients[provider_id]

        config = self.providers[provider_id]
        api_key = self._get_api_key(provider_id)
        if not api_key:
            return None

        sdk = config.get("sdk", "openai")

        if sdk == "anthropic":
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                result = (client, True)
            except ImportError:
                print(
                    f"[llm-router] Anthropic SDK not installed for {provider_id}, skipping",
                    file=sys.stderr,
                )
                return None
        else:
            from openai import OpenAI

            client = OpenAI(
                base_url=config.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
            )
            result = (client, False)

        self._clients[provider_id] = result
        return result

    def _call_provider(
        self,
        provider_id: str,
        messages: list[dict],
        **kwargs,
    ) -> RouterResult:
        """Make an API call to a specific provider.

        Handles SDK differences between OpenAI-compatible and Anthropic.
        """
        config = self.providers[provider_id]
        model = kwargs.pop("model", None) or config.get("default_model")

        client_info = self._get_client(provider_id)
        if client_info is None:
            raise ConnectionError(f"No client available for {provider_id}")

        client, is_anthropic = client_info

        # Extract common params
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)

        if is_anthropic:
            # Anthropic SDK: extract system message if present
            system_msg = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)

            call_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages,
            }
            if system_msg:
                call_kwargs["system"] = system_msg

            response = client.messages.create(**call_kwargs)
            text = response.content[0].text
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }
        else:
            # OpenAI-compatible SDK
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                }

        return RouterResult(
            text=text,
            provider_id=provider_id,
            provider_name=config.get("name", provider_id),
            model=model,
            usage=usage,
        )

    def _is_rate_limit_error(self, error: Exception) -> tuple[bool, int | None]:
        """Check if an error is a rate limit error.

        Returns (is_rate_limit, retry_after_seconds).
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for HTTP 429
        if "429" in error_str or "rate" in error_str and "limit" in error_str:
            # Try to extract Retry-After
            retry_after = None
            match = re.search(r"retry.?after[:\s]+(\d+)", error_str, re.IGNORECASE)
            if match:
                retry_after = int(match.group(1))

            # Check for response headers if available
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                headers = error.response.headers
                if "retry-after" in headers:
                    try:
                        retry_after = int(headers["retry-after"])
                    except (ValueError, TypeError):
                        pass

            return True, retry_after

        # Check for quota exceeded
        if "quota" in error_str or "insufficient" in error_str:
            return True, None

        return False, None

    def call(
        self,
        profile: str,
        messages: list[dict],
        **kwargs,
    ) -> RouterResult:
        """Route an LLM request through the priority cascade.

        Args:
            profile: Name of the routing profile (e.g. "free-only", "paid-first").
            messages: List of message dicts (OpenAI format: role + content).
            **kwargs: Additional params passed to the API (temperature, max_tokens, etc.).

        Returns:
            RouterResult with the response text and metadata.

        Raises:
            AllProvidersExhausted: If no provider could fulfill the request.
            KeyError: If the profile doesn't exist.
        """
        if profile not in self.profiles:
            raise KeyError(
                f"Unknown profile '{profile}'. Available: {list(self.profiles.keys())}"
            )

        provider_ids = self.profiles[profile].get("providers", [])
        attempts = []

        # Clear expired strikes before routing
        self.tracker.clear_expired()

        for pid in provider_ids:
            if pid not in self.providers:
                print(
                    f"[llm-router] Skipping unknown provider '{pid}'", file=sys.stderr
                )
                continue

            # Check if API key is available
            if not self._get_api_key(pid):
                attempts.append(
                    {"provider": pid, "status": "skipped", "reason": "no API key"}
                )
                continue

            # Check if provider is struck
            if not self.tracker.is_available(pid):
                strike = self.tracker.strikes.get(pid, {})
                attempts.append(
                    {
                        "provider": pid,
                        "status": "struck",
                        "reason": strike.get("reason", "rate limited"),
                        "renews_at": strike.get("renews_at"),
                    }
                )
                print(
                    f"[llm-router] Skipping {pid} (struck until {strike.get('renews_at')})",
                    file=sys.stderr,
                )
                continue

            # Try this provider
            try:
                config = self.providers[pid]
                print(
                    f"[llm-router] Trying {config.get('name', pid)} ({config.get('default_model', '?')})",
                    file=sys.stderr,
                )
                result = self._call_provider(pid, messages, **kwargs)
                print(
                    f"[llm-router] Success: {result.provider_name} ({result.model})",
                    file=sys.stderr,
                )
                return result

            except Exception as e:
                is_rate_limit, retry_after = self._is_rate_limit_error(e)
                error_reason = f"{type(e).__name__}: {str(e)[:200]}"

                if is_rate_limit:
                    renewal = self.providers[pid].get("rate_limit", {}).get(
                        "renewal", "rolling"
                    )
                    self.tracker.strike(
                        pid,
                        reason=error_reason,
                        renewal_policy=renewal,
                        retry_after_seconds=retry_after,
                    )
                    print(
                        f"[llm-router] Rate limited by {pid}, struck. Trying next...",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[llm-router] Error from {pid}: {error_reason}",
                        file=sys.stderr,
                    )

                attempts.append(
                    {
                        "provider": pid,
                        "status": "failed",
                        "reason": error_reason,
                        "rate_limited": is_rate_limit,
                    }
                )
                continue

        raise AllProvidersExhausted(attempts)

    def list_available(self, profile: str) -> list[dict]:
        """List available providers for a profile (with API key + not struck).

        Useful for debugging which providers are currently active.
        """
        if profile not in self.profiles:
            return []

        self.tracker.clear_expired()
        result = []
        for pid in self.profiles[profile].get("providers", []):
            if pid not in self.providers:
                continue
            config = self.providers[pid]
            has_key = bool(self._get_api_key(pid))
            is_available = self.tracker.is_available(pid)
            result.append(
                {
                    "id": pid,
                    "name": config.get("name", pid),
                    "model": config.get("default_model"),
                    "tier": config.get("tier"),
                    "has_key": has_key,
                    "is_available": is_available and has_key,
                    "strike": self.tracker.strikes.get(pid),
                }
            )
        return result
