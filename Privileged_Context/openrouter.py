"""Cached OpenRouter requests with pacing, retries, and optional sampled images."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib import error, request
from urllib.parse import urlsplit

from .frames import frame_blocks

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
_LOCK = threading.Lock()
_LAST_START = 0.0
_TRANSIENT_TYPES = {'rate_limit_exceeded', 'provider_overloaded', 'provider_unavailable',
                    'server', 'timeout', 'unmapped'}
_PERMANENT_TYPES = {'context_length_exceeded', 'max_tokens_exceeded', 'token_limit_exceeded',
                    'string_too_long', 'authentication', 'permission_denied', 'payment_required',
                    'invalid_request', 'invalid_prompt', 'not_found', 'precondition_failed',
                    'payload_too_large', 'unprocessable', 'content_policy_violation', 'refusal',
                    'invalid_image', 'image_too_large', 'image_too_small', 'unsupported_image_format',
                    'image_not_found'}


def response_error(response):
    """Find errors reported after the provider has already sent HTTP headers."""
    if response.get('error'):
        return response['error']
    for choice in response.get('choices') or []:
        if choice.get('error'):
            return choice['error']
        if choice.get('finish_reason') == 'error':
            return {'metadata': {'error_type': 'unmapped'}}
    return None


def classify_response_error(problem):
    """Only allowlisted categories and numeric codes may reach logs/exceptions."""
    problem = problem if isinstance(problem, dict) else {}
    code = problem.get('code')
    if isinstance(code, str) and code.isascii() and code.isdigit() and len(code) == 3:
        code = int(code)
    if type(code) is not int or not 100 <= code <= 599:
        code = None
    metadata = problem.get('metadata')
    kind = metadata.get('error_type') if isinstance(metadata, dict) else None
    if not isinstance(kind, str) or kind not in _TRANSIENT_TYPES | _PERMANENT_TYPES:
        kind = 'unmapped'
    if kind in _PERMANENT_TYPES:
        retryable = False
    elif kind != 'unmapped':
        retryable = True
    elif code is not None and 400 <= code < 500:
        retryable = code in (408, 429)
    else:
        # Unknown provider failures get bounded retries, rather than stopping immediately.
        retryable = True
    return code, kind, retryable


def retry_delay(headers, default):
    try:
        return max(default, min(300, float(headers.get('Retry-After', 0))))
    except (AttributeError, ValueError, TypeError):
        return default


def record_response_error(cache, identity, attempt, maximum, code, kind, retryable, delay):
    event = {'request_sha256': identity, 'attempt': attempt, 'max_attempts': maximum,
             'error_code': code, 'error_type': kind, 'retryable': retryable,
             'retrying': retryable and attempt < maximum}
    event['retry_delay_seconds'] = delay if event['retrying'] else None
    path = Path(cache) / 'transport_errors.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')
    print(f'OpenRouter response error: code={code} type={kind}; '
          f'attempt {attempt}/{maximum}; retrying={event["retrying"]}', file=sys.stderr, flush=True)


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'https' or parsed.hostname != 'openrouter.ai' or
            parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('OpenRouter credentials may only be sent to https://openrouter.ai')


def check_free_route(model, timeout=30, allow_paid=False):
    """Check current routing/prices; never silently substitute a paid model."""
    with request.urlopen('https://openrouter.ai/api/v1/models', timeout=timeout) as response:
        models = json.load(response)['data']
    matches = [m for m in models if m['id'] == model]
    if len(matches) != 1:
        raise ValueError(f'Exact model route is unavailable: {model}')
    entry = matches[0]
    prices = entry.get('pricing', {})
    if (not allow_paid or model.endswith(':free')) and any(float(prices.get(k, 'nan')) != 0 for k in ('prompt', 'completion')):
        raise ValueError(f'Free route pricing check failed: {model}')
    return {'id': entry['id'], 'pricing': prices,
            'input_modalities': entry.get('architecture', {}).get('input_modalities', [])}


def payload_for(messages, args, record=None):
    actual = deepcopy(messages)
    if args.include_frames and record is not None:
        blocks = frame_blocks(record)
        converted = []
        for block in blocks:
            if block['type'] == 'image':
                source = block['source']
                converted.append({'type': 'image_url', 'image_url': {
                    'url': 'data:' + source['media_type'] + ';base64,' + source['data']}})
            else:
                converted.append(block)
        if not actual or actual[1]['role'] != 'user' or not isinstance(actual[1]['content'], str):
            raise ValueError('Frame requests require an initial text user message')
        actual[1]['content'] = [{'type': 'text', 'text': actual[1]['content']}, *converted]
    payload = {'model': args.model, 'messages': actual, 'max_tokens': args.max_tokens}
    if args.temperature is not None:
        payload['temperature'] = args.temperature
    return payload


def call_model(messages, args, cache, record=None):
    from .util import sha, read, write
    global _LAST_START
    validate_endpoint(args.endpoint)
    payload = payload_for(messages, args, record)
    identity = sha({'endpoint': args.endpoint, 'payload': payload})
    path = Path(cache) / (identity + '.json')
    if path.exists():
        saved = read(path)
        if saved['request_sha256'] != identity:
            raise ValueError('Cache identity mismatch')
        response = saved['response']
    else:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError(f'Set {args.api_key_env} privately in the job environment')
        req = request.Request(args.endpoint, data=json.dumps(payload).encode(), headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        for attempt in range(args.max_attempts):
            with _LOCK:
                time.sleep(max(0, args.request_interval - (time.monotonic() - _LAST_START)))
                _LAST_START = time.monotonic()
            delay = min(120, 2 ** attempt)
            try:
                with request.urlopen(req, timeout=args.timeout) as stream:
                    response = json.load(stream)
                    delay = retry_delay(getattr(stream, 'headers', None), delay)
                problem = response_error(response)
                if problem is not None:
                    code, kind, retryable = classify_response_error(problem)
                    record_response_error(cache, identity, attempt + 1, args.max_attempts,
                                          code, kind, retryable, delay)
                    if not retryable or attempt + 1 == args.max_attempts:
                        raise RuntimeError(f'OpenRouter provider error: code={code} type={kind}; '
                                           f'attempt {attempt + 1}/{args.max_attempts}; no fallback requested')
                    time.sleep(delay)
                    continue
                break
            except error.HTTPError as exc:
                # Provider error bodies may echo inputs; do not put credentials or raw bodies in logs.
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == args.max_attempts:
                    raise RuntimeError(f'OpenRouter HTTP {exc.code}; no fallback requested') from None
                delay = retry_delay(exc.headers, delay)
            except (error.URLError, TimeoutError):
                if attempt + 1 == args.max_attempts:
                    raise RuntimeError('OpenRouter transport failed') from None
            time.sleep(delay)
        write(path, {'request_sha256': identity, 'response': response})
    choices = response.get('choices') or []
    if len(choices) != 1 or choices[0].get('finish_reason') not in ('stop', 'end_turn'):
        raise ValueError('Incomplete/truncated response; inspect cached response')
    text = choices[0].get('message', {}).get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Model returned no visible text')
    return text
