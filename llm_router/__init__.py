"""LLM Router — Lightweight API routing with priority cascading and rate limit tracking.

Usage:
    from llm_router import LLMRouter

    router = LLMRouter()
    result = router.call("free-only", messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(result.text)
"""

from .router import AllProvidersExhausted, LLMRouter, RouterResult

__version__ = "0.1.0"
__all__ = ["LLMRouter", "RouterResult", "AllProvidersExhausted"]


def call_llm(profile: str, messages: list[dict], **kwargs) -> RouterResult:
    """One-shot convenience function.

    Creates a router with default config and makes a single call.
    For repeated calls, create an LLMRouter instance directly.
    """
    router = LLMRouter()
    return router.call(profile, messages, **kwargs)
