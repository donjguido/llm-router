"""Configuration loading for LLM Router.

Loads provider definitions and routing profiles from YAML files.
Supports layered config: bundled defaults + user overrides.
"""

from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_PROVIDERS = PACKAGE_ROOT / "config" / "providers.yml"
DEFAULT_PROFILES = PACKAGE_ROOT / "config" / "profiles.yml"


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_providers(path: str | Path | None = None) -> dict[str, dict]:
    """Load provider definitions.

    Args:
        path: Custom providers.yml path. If None, uses bundled defaults.
              If provided, merges with defaults (custom overrides defaults).

    Returns:
        Dict mapping provider_id -> provider config.
    """
    providers = load_yaml(DEFAULT_PROVIDERS).get("providers", {})

    if path:
        custom_path = Path(path)
        if custom_path.exists():
            custom = load_yaml(custom_path).get("providers", {})
            # Deep merge: custom overrides defaults per provider
            for pid, pconfig in custom.items():
                if pid in providers:
                    providers[pid].update(pconfig)
                else:
                    providers[pid] = pconfig

    return providers


def load_profiles(path: str | Path | None = None) -> dict[str, dict]:
    """Load routing profiles.

    Args:
        path: Custom profiles.yml path. If None, uses bundled defaults.
              If provided, merges with defaults (custom overrides defaults).

    Returns:
        Dict mapping profile_name -> profile config.
    """
    profiles = load_yaml(DEFAULT_PROFILES).get("profiles", {})

    if path:
        custom_path = Path(path)
        if custom_path.exists():
            custom = load_yaml(custom_path).get("profiles", {})
            profiles.update(custom)

    return profiles


def validate_config(providers: dict, profiles: dict) -> list[str]:
    """Validate that all profiles reference valid providers.

    Returns list of warning messages (empty if all valid).
    """
    warnings = []
    provider_ids = set(providers.keys())

    for profile_name, profile in profiles.items():
        for pid in profile.get("providers", []):
            if pid not in provider_ids:
                warnings.append(
                    f"Profile '{profile_name}' references unknown provider '{pid}'"
                )

    for pid, pconfig in providers.items():
        if "env_key" not in pconfig:
            warnings.append(f"Provider '{pid}' missing 'env_key'")
        if "sdk" not in pconfig:
            warnings.append(f"Provider '{pid}' missing 'sdk'")
        if pconfig.get("sdk") not in ("openai", "anthropic"):
            warnings.append(
                f"Provider '{pid}' has unsupported sdk '{pconfig.get('sdk')}'"
            )

    return warnings
