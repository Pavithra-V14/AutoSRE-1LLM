from __future__ import annotations

import time
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")


def post_with_retry(
    call: Callable[[], httpx.Response],
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> httpx.Response:
    """Retries `call` (expected to perform an httpx POST and return the
    Response) on httpx.HTTPError, with exponential backoff. Raises the
    last exception if all attempts fail."""
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = call()
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(base_delay_seconds * (2 ** attempt))

    assert last_error is not None
    raise last_error
