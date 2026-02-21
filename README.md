# LLM Router

**Lightweight LLM API router with priority cascading, rate limit tracking, and model scouting.**

Designed for CI/CD pipelines and GitHub Actions. No servers, no databases — just YAML config and JSON state files.

---

## Features

- **Priority Cascading** — Define ordered lists of API providers. The router tries each in order, falling back automatically on failure.
- **Rate Limit Tracking** — When a provider hits a rate limit (HTTP 429), it's "struck" and skipped until its renewal time.
- **SDK Abstraction** — Handles both OpenAI-compatible and Anthropic APIs transparently. Callers use one interface.
- **File-Based Config** — YAML provider definitions and routing profiles. Fork-friendly and CI/CD-native.
- **Model Scout** — Weekly GitHub Action searches the web (via BraveAPI) for the best free models and flags updates.
- **Layered Overrides** — Bundled defaults + per-repo custom config. Downstream repos can override specific providers or profiles.

## Quick Start

### Install

```bash
pip install git+https://github.com/donjguido/llm-router.git
```

With Anthropic support:
```bash
pip install "llm-router[anthropic] @ git+https://github.com/donjguido/llm-router.git"
```

### Usage

```python
from llm_router import LLMRouter

# Uses bundled default config
router = LLMRouter()

# Route through free providers (Gemini > OpenRouter > Mistral)
result = router.call("free-only", messages=[
    {"role": "user", "content": "Explain quantum computing in one sentence."}
], temperature=0.7, max_tokens=200)

print(result.text)
print(f"Served by: {result.provider_name} ({result.model})")
```

### With Custom Config

```python
router = LLMRouter(
    providers_file="my-providers.yml",   # Override/extend defaults
    profiles_file="my-profiles.yml",     # Add custom profiles
    tracker_file="tracker-state.json",   # Persist rate limit strikes
)

result = router.call("my-custom-profile", messages=[...])
```

### One-Shot Convenience

```python
from llm_router import call_llm

result = call_llm("paid-first", messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
])
```

## Configuration

### Providers (`config/providers.yml`)

Each provider defines an API endpoint:

```yaml
providers:
  gemini-free:
    name: "Google Gemini (Free)"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
    env_key: "GEMINI_API_KEY"          # Environment variable for the API key
    sdk: "openai"                       # "openai" or "anthropic"
    default_model: "gemini-2.0-flash"
    tier: "free"                        # "free" or "paid"
    rate_limit:
      requests_per_minute: 15
      renewal: "rolling"                # "rolling", "daily", or "monthly"
```

### Profiles (`config/profiles.yml`)

Profiles are ordered lists of providers:

```yaml
profiles:
  free-only:
    description: "Only free-tier APIs"
    providers: [gemini-free, openrouter-free, mistral-free]

  paid-first:
    description: "Best quality first"
    providers: [anthropic, openai, grok, gemini-free, openrouter-free]
```

### Bundled Providers

| Provider | Tier | Default Model | SDK |
|----------|------|---------------|-----|
| Google Gemini | Free | gemini-2.0-flash | openai |
| OpenRouter | Free | stepfun/step-3.5-flash:free | openai |
| Mistral | Free | mistral-small-latest | openai |
| Anthropic Claude | Paid | claude-sonnet-4-20250514 | anthropic |
| OpenAI | Paid | gpt-4o | openai |
| xAI Grok | Paid | grok-3-mini | openai |

## Environment Variables

Set API keys as environment variables. The router only tries providers whose keys are present:

```bash
export GEMINI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export XAI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
```

## Rate Limit Tracking

When a provider returns HTTP 429 or a quota error, the router:

1. **Strikes** the provider with a renewal timestamp
2. **Saves** the strike to `tracker-state.json` (if configured)
3. **Skips** the provider on subsequent calls until renewal
4. **Auto-clears** expired strikes

```python
# Check what's available
for p in router.list_available("free-only"):
    status = "available" if p["is_available"] else "unavailable"
    print(f"{p['name']}: {status}")
```

## Model Scout

The model scout runs weekly (or on-demand) to search the web for current free model availability.

### Setup

1. Get a [BraveAPI key](https://brave.com/search/api/) (free tier: 2,000 queries/month)
2. Add `BRAVE_API_KEY` as a GitHub secret
3. The workflow runs every Monday at 9am UTC

### Manual Run

```bash
BRAVE_API_KEY=your-key python -m llm_router.scout
```

## GitHub Actions Integration

```yaml
- name: Install LLM Router
  run: pip install git+https://github.com/donjguido/llm-router.git

- name: Generate content
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: python my_script.py
```

## Forking

This repo is designed to be forked:

1. Fork the repo
2. Add your API keys as GitHub secrets
3. Customize `config/providers.yml` and `config/profiles.yml`
4. The model scout workflow will run automatically

To stay updated with upstream changes, periodically sync your fork.

## License

MIT
