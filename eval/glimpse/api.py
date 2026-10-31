"""OpenRouter JSON transport adapted from the local Claude/Eulorian evaluators."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import hashlib
import math
from pathlib import Path
import random
import re
import stat
import time
import urllib.error
import urllib.request


def read_key(path):
    path = Path(path)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Credential file must be owner-only")
    key = path.read_text().strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Invalid credential file")
    return key


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 10
    max_elapsed_seconds: float = 1800
    timeout_seconds: float = 600
    initial_delay_seconds: float = 2
    max_delay_seconds: float = 60

    def __post_init__(self):
        if self.max_attempts < 1 or any(not math.isfinite(v) or v <= 0 for v in (
                self.max_elapsed_seconds, self.timeout_seconds,
                self.initial_delay_seconds, self.max_delay_seconds)):
            raise ValueError("API retry counts and time limits must be positive and finite")


class APIRequestError(RuntimeError):
    """A permanent API error or exhausted transient-error retry budget."""
    def __init__(self, status, attempts, retryable):
        self.status, self.attempts, self.retryable = status, attempts, retryable
        reason = "retry budget exhausted" if retryable else "non-retryable error"
        super().__init__(f"OpenRouter {status or 'response/transport error'}: {reason} after {attempts} attempt(s)")


def add_retry_arguments(parser):
    defaults = RetryPolicy()
    parser.add_argument('--api-max-attempts', type=int, default=defaults.max_attempts)
    parser.add_argument('--api-retry-seconds', type=float, default=defaults.max_elapsed_seconds,
                        help='Total retry time budget per model turn, including requests and backoff')
    parser.add_argument('--api-timeout-seconds', type=float, default=defaults.timeout_seconds)


def retry_policy_from_args(args):
    return RetryPolicy(max_attempts=args.api_max_attempts,
                       max_elapsed_seconds=args.api_retry_seconds,
                       timeout_seconds=args.api_timeout_seconds)


def _safe(value, key):
    """Bound provider diagnostics and remove credentials or reflected image payloads."""
    if isinstance(value, dict):
        return {str(k): ('[redacted]' if re.search(r'authorization|api.?key|token_secret', str(k), re.I)
                         else _safe(v, key)) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v, key) for v in value[:20]]
    if isinstance(value, str):
        value = value.replace(key, '[redacted]') if key else value
        value = re.sub(r'data:image/[^\s"\']+', '[image payload omitted]', value)
        value = re.sub(r'(?i)Bearer\s+[^\s"\']+|sk-[A-Za-z0-9_-]+', '[redacted]', value)
        return value[:4096]
    return value


def _retry_after(headers):
    value = headers.get('Retry-After')
    if value is None:
        return 0
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


TRANSIENT_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
TRANSIENT_TYPES = {'server', 'server_error', 'internal_server_error', 'rate_limit_exceeded',
                   'provider_overloaded', 'provider_unavailable', 'timeout', 'request_timeout'}
PERMANENT_TYPES = {'invalid_request', 'invalid_request_error', 'invalid_prompt', 'authentication_error',
                   'invalid_api_key', 'permission_denied', 'insufficient_credits', 'insufficient_quota',
                   'context_length_exceeded', 'not_found', 'content_filter', 'moderation_blocked'}


def _transient(error, status):
    error = error if isinstance(error, dict) else {}
    metadata = error.get('metadata') or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    if status == 402:
        return metadata.get('limit_source') == 'openrouter_in_flight_budget'
    if status is not None and 400 <= status < 500 and status not in TRANSIENT_CODES:
        return False
    kind = metadata.get('error_type') or error.get('code')
    if isinstance(kind, str) and kind in PERMANENT_TYPES:
        return False
    return status in TRANSIENT_CODES or (isinstance(kind, str) and kind in TRANSIENT_TYPES) or status is None


def _error_status(error, default=None):
    try:
        return int(error.get('code', default))
    except (TypeError, ValueError, AttributeError):
        return default


def request(payload, key, *, policy=None, on_attempt=None):
    """Retry one immutable model request; never advance chat history on a failed attempt."""
    policy = policy or RetryPolicy()
    data = json.dumps(payload).encode()
    identity = {'request_bytes': len(data), 'request_sha256': hashlib.sha256(data).hexdigest(),
                'image_count': sum(p.get('type') == 'image_url' for m in payload.get('messages', [])
                    if isinstance(m.get('content'), list) for p in m['content'] if isinstance(p, dict))}
    started = time.monotonic()
    for attempt in range(1, policy.max_attempts + 1):
        attempt_started = time.monotonic()
        remaining = policy.max_elapsed_seconds - (attempt_started - started)
        if remaining <= 0:
            raise APIRequestError(None, attempt - 1, True)
        event = {'attempt': attempt, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                 **identity, 'request_timeout_seconds': min(policy.timeout_seconds, remaining)}
        result, status, headers, error = None, None, {}, None
        http_status = None
        retryable = True
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "X-OpenRouter-Title": "SPOT Modified EVA Glimpse"})
        try:
            with urllib.request.urlopen(req, timeout=min(policy.timeout_seconds, remaining)) as response:
                status = response.status
                http_status = status
                headers = response.headers or {}
                result = json.load(response)
            if not isinstance(result, dict):
                error = {'message': 'Invalid JSON response shape'}
            else:
                choices = result.get('choices')
                choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else {}
                choice = choice if isinstance(choice, dict) else {}
                error = result.get('error') or choice.get('error')
                if error or choice.get('finish_reason') == 'error':
                    error = error or {'message': 'Provider returned finish_reason=error'}
                    status = _error_status(error)
                    retryable = _transient(error, status)
                elif not isinstance(choice.get('message'), dict):
                    error = {'message': 'Incomplete API response'}
        except urllib.error.HTTPError as exc:
            status, headers = exc.code, exc.headers or {}
            http_status = status
            try:
                try:
                    body = exc.read(65536).decode('utf-8', errors='replace')
                except (OSError, http.client.HTTPException) as read_error:
                    body = f'Could not read HTTP error body: {type(read_error).__name__}'
                try:
                    result = json.loads(body)
                except ValueError:
                    result = {'error': {'message': body}}
                error = (result.get('error') or result) if isinstance(result, dict) else {'message': body}
            finally:
                exc.close()
            retryable = _transient(error, status)
        except (OSError, http.client.HTTPException,
                json.JSONDecodeError, UnicodeDecodeError) as exc:
            error = {'message': str(exc), 'type': type(exc).__name__}
        event.update(http_status=http_status, api_status=status, duration_seconds=time.monotonic() - attempt_started,
                     elapsed_seconds=time.monotonic() - started)
        event['response_headers'] = {k: headers.get(k) for k in (
            'Retry-After', 'X-Request-Id', 'X-Generation-Id', 'CF-Ray') if headers.get(k)}
        if isinstance(result, dict):
            event.update(generation_id=result.get('id'), provider=result.get('provider'),
                         usage=result.get('usage'))
        if error is None:
            event['outcome'] = 'succeeded'
            if on_attempt:
                on_attempt(_safe(event, key))
            return result
        event.update(error=error, retryable=retryable)
        if isinstance(result, dict) and result.get('choices'):
            event['partial_response'] = result['choices']
        delay = max(_retry_after(headers), random.uniform(0.5, 1.0) * min(
            policy.max_delay_seconds, policy.initial_delay_seconds * 2 ** min(attempt - 1, 30)))
        remaining = policy.max_elapsed_seconds - (time.monotonic() - started)
        retry = retryable and attempt < policy.max_attempts and delay < remaining
        event.update(outcome='retry' if retry else 'failed', retry_delay_seconds=delay if retry else None)
        if on_attempt:
            on_attempt(_safe(event, key))
        if not retry:
            raise APIRequestError(status, attempt, retryable) from None
        time.sleep(delay)
