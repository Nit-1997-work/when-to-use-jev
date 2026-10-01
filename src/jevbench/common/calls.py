"""Result types shared by every system under test, whatever the experiment."""

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    OK = "ok"
    # An answer that breaks the expected shape (unparseable, outside the allowed values). It counts as wrong.
    INVALID_OUTPUT = "invalid_output"
    # Blocked: the provider's safety filter withheld the answer. Excluded from quality metrics and reported separately.
    # The stored value stays "refused" so earlier result files keep their meaning.
    REFUSED = "refused"
    # Non-retryable request failure, or retries exhausted. Excluded from quality metrics; a resumed run redoes it.
    API_ERROR = "api_error"


class RetryableError(Exception):
    """Transient failure (rate limit, 408/409, 5xx, timeout, connection). The runner retries and records attempts."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        # Server-requested wait (Retry-After), honoured by the runner's backoff when present.
        self.retry_after_s = retry_after_s
