"""Strike tracking for rate-limited providers.

Persists strike data to a JSON file so state carries across CI/CD runs.
When a provider hits a rate limit, it's "struck" with a renewal time.
The tracker automatically clears expired strikes on each check.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class StrikeTracker:
    """Track rate limit strikes for API providers."""

    def __init__(self, state_file: str | Path | None = None):
        """Initialize tracker.

        Args:
            state_file: Path to JSON file for persisting state.
                       If None, operates in-memory only.
        """
        self.state_file = Path(state_file) if state_file else None
        self.strikes: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load state from file."""
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.strikes = data.get("strikes", {})
            except (json.JSONDecodeError, KeyError):
                self.strikes = {}

    def _save(self):
        """Persist state to file."""
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"strikes": self.strikes}
            self.state_file.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )

    def is_available(self, provider_id: str) -> bool:
        """Check if a provider is available (not struck, or strike expired).

        Automatically clears expired strikes.
        """
        if provider_id not in self.strikes:
            return True

        strike = self.strikes[provider_id]
        renews_at = strike.get("renews_at")

        if renews_at:
            renewal_time = datetime.fromisoformat(renews_at)
            if datetime.now(timezone.utc) >= renewal_time:
                # Strike expired, clear it
                del self.strikes[provider_id]
                self._save()
                return True

        return False

    def strike(
        self,
        provider_id: str,
        reason: str,
        renewal_policy: str = "rolling",
        retry_after_seconds: int | None = None,
    ):
        """Mark a provider as struck (rate-limited).

        Args:
            provider_id: Provider to strike.
            reason: Why it was struck (e.g. "429 Too Many Requests").
            renewal_policy: "rolling" (retry after cooldown), "daily" (midnight UTC),
                           "monthly" (start of next month).
            retry_after_seconds: If server provided Retry-After header, use that.
        """
        now = datetime.now(timezone.utc)

        if retry_after_seconds:
            renews_at = now + timedelta(seconds=retry_after_seconds)
        elif renewal_policy == "daily":
            # Renew at midnight UTC
            tomorrow = now.date() + timedelta(days=1)
            renews_at = datetime(
                tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc
            )
        elif renewal_policy == "monthly":
            # Renew at start of next month
            if now.month == 12:
                renews_at = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                renews_at = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
        else:
            # Rolling: default 60-second cooldown
            renews_at = now + timedelta(seconds=60)

        self.strikes[provider_id] = {
            "struck_at": now.isoformat(),
            "reason": reason,
            "renews_at": renews_at.isoformat(),
        }
        self._save()

    def clear_expired(self):
        """Remove all expired strikes."""
        now = datetime.now(timezone.utc)
        expired = []
        for pid, strike in self.strikes.items():
            renews_at = strike.get("renews_at")
            if renews_at:
                if datetime.now(timezone.utc) >= datetime.fromisoformat(renews_at):
                    expired.append(pid)

        for pid in expired:
            del self.strikes[pid]

        if expired:
            self._save()

        return expired

    def get_status(self) -> dict[str, Any]:
        """Return current strike status for all providers."""
        self.clear_expired()
        return {
            "active_strikes": len(self.strikes),
            "strikes": dict(self.strikes),
        }
