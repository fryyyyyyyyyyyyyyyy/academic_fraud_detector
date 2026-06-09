"""
Rate-limited API client with retry logic.

Used by all tools that communicate with external APIs (arXiv, CrossRef,
Semantic Scholar, etc.) to ensure we stay within rate limits and handle
transient failures gracefully.
"""

import time
import logging
from typing import Optional, Callable
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        """Block until it's safe to make the next call."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


# Global rate limiters per API
_rate_limiters: dict[str, RateLimiter] = {
    "arxiv": RateLimiter(calls_per_second=0.5),            # arXiv free: be conservative
    "crossref": RateLimiter(calls_per_second=3.0),         # CrossRef: polite rate
    "semantic_scholar": RateLimiter(calls_per_second=1.0), # S2 free tier: 1 req/s max
}


def get_session(
    total_retries: int = 5,
    backoff_factor: float = 2.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Create a requests.Session with retry configuration."""
    session = requests.Session()
    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def rate_limited(api_name: str) -> Callable:
    """Decorator to rate-limit a function by API name."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = _rate_limiters.get(api_name)
            if limiter:
                limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_request(
    url: str,
    method: str = "GET",
    timeout: int = 30,
    api_name: Optional[str] = None,
    **kwargs,
) -> requests.Response:
    """
    Make an HTTP request with rate limiting, retries, and error handling.

    Args:
        url: The URL to request.
        method: HTTP method.
        timeout: Request timeout in seconds.
        api_name: Name of the API for rate limiting ('arxiv', 'crossref', 'semantic_scholar').
        **kwargs: Additional arguments passed to requests.

    Returns:
        requests.Response object.

    Raises:
        requests.RequestException: On non-retryable errors.
    """
    if api_name and api_name in _rate_limiters:
        _rate_limiters[api_name].wait()

    session = get_session()
    try:
        response = session.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        raise
