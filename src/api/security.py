"""API key authentication and per-key request quota controls."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
import time
from typing import Deque, Dict, Optional

from fastapi import Request

from src.api.config import AppSettings


@dataclass
class AuthQuotaDecision:
    """Authentication + quota evaluation for one request."""

    allowed: bool
    status_code: int
    detail: str = ""
    limit: int = 0
    remaining: int = 0
    retry_after_seconds: int = 0


class ApiSecurityManager:
    """Thread-safe API auth and sliding-window quota manager."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._quota_lock = Lock()
        self._quota_buckets: Dict[str, Deque[float]] = {}

    def is_protected_path(self, path: str) -> bool:
        if not path.startswith("/api/"):
            return False
        return path not in set(self._settings.api_auth_exempt_paths)

    def evaluate(self, request: Request) -> Optional[AuthQuotaDecision]:
        """Validate auth/quota for request or return None when not applicable."""
        if not self._settings.api_auth_enabled:
            return None
        if not self.is_protected_path(request.url.path):
            return None

        key_header = self._settings.api_key_header
        raw_key = request.headers.get(key_header, "").strip()
        configured_keys = set(self._settings.api_keys)
        if not configured_keys:
            return AuthQuotaDecision(
                allowed=False,
                status_code=503,
                detail="API authentication is enabled but no API keys are configured.",
            )

        if not raw_key or raw_key not in configured_keys:
            return AuthQuotaDecision(
                allowed=False,
                status_code=401,
                detail=f"Missing or invalid API key. Provide '{key_header}' header.",
            )

        limit = int(self._settings.api_quota_per_minute)
        now = time.monotonic()
        window_start = now - 60.0

        with self._quota_lock:
            bucket = self._quota_buckets.get(raw_key)
            if bucket is None:
                bucket = deque()
                self._quota_buckets[raw_key] = bucket

            while bucket and bucket[0] <= window_start:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(round(60.0 - (now - bucket[0]))))
                return AuthQuotaDecision(
                    allowed=False,
                    status_code=429,
                    detail="API quota exceeded. Try again later.",
                    limit=limit,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return AuthQuotaDecision(
                allowed=True,
                status_code=200,
                limit=limit,
                remaining=remaining,
            )
