"""Cached Claude Code CLI transport, following claude_eval's isolated -p calls."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .frames import frame_blocks


_START_LOCK = threading.Lock()
_LAST_START = 0.0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _conversation(messages):
    if not messages or messages[0]["role"] != "system":
        raise ValueError("Claude request requires a system prompt first")
    if len(messages) == 2 and messages[1]["role"] == "user" and isinstance(messages[1]["content"], str):
        return messages[0]["content"], messages[1]["content"]
    chunks = []
    for i, message in enumerate(messages[1:]):
        role = message["role"]
        if role not in ("user", "assistant") or not isinstance(message["content"], str):
            raise ValueError("Only text user/assistant correction turns are supported")
        label = "Trajectory input" if i == 0 else ("Previous assistant response" if role == "assistant" else "Validation feedback")
        chunks.append(f"[{label}]\n{message['content']}")
    if not chunks:
        raise ValueError("Claude request has no user content")
    return messages[0]["content"], "\n\n".join(chunks)


def command(args, system, with_frames):
    cmd = [args.claude_bin, "-p", "--model", args.model,
           "--system-prompt", system, "--tools", "", "--strict-mcp-config",
           "--permission-mode", "dontAsk", "--setting-sources", "",
           "--disable-slash-commands", "--no-session-persistence",
           "--max-turns", "2", "--output-format", "stream-json", "--verbose"]
    if with_frames:
        cmd.extend(["--input-format", "stream-json"])
    return cmd


def _strip_images(value):
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url"):
            return {"type": "image", "omitted": True}
        return {k: _strip_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    return value


def _invoke(cmd, payload, cwd, env, trace_path, stderr_path, timeout):
    result = None
    with trace_path.open("w", encoding="utf-8") as trace, stderr_path.open("w", encoding="utf-8") as stderr:
        process_options = ({"start_new_session": True} if os.name == "posix"
                           else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP})
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr, text=True, encoding="utf-8", cwd=cwd, env=env,
                                **process_options)
        def stop():
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                pass
        try:
            try:
                output, _ = proc.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                stop()
                proc.communicate()
                raise TimeoutError("Claude Code call timed out") from None
            for line in output.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    raise RuntimeError("Claude emitted a non-JSON stream") from None
                trace.write(json.dumps(_strip_images(event), ensure_ascii=False) + "\n")
                if event.get("type") == "result":
                    result = event
            code = proc.returncode
            if code or result is None or result.get("is_error"):
                subtype = result.get("subtype") if result else "no_result"
                raise RuntimeError(f"Claude Code failed (exit {code}, {subtype}); inspect stderr/trace")
            answer = result.get("result")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Claude returned no visible text")
            return result
        finally:
            if proc.poll() is None:
                stop()
                proc.communicate()


def call_model(messages, args, cache, record=None):
    """Send one Claude Code conversation; only memory calls may attach frames."""
    global _LAST_START
    system, user = _conversation(messages)
    include_frames = bool(getattr(args, "include_frames", False) and record is not None)
    identity = _digest({"model": args.model, "claude_version": args.claude_version,
                        "messages": messages,
                        "frames": record["trajectory_sha256"] if include_frames else None,
                        "transport": "claude-code-stream-json" if include_frames else "claude-code-text"})
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / f"{identity}.json"
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        if data.get("request_sha256") != identity:
            raise ValueError("Claude cache identity mismatch")
        return data["text"]
    if include_frames:
        blocks = [{"type": "text", "text": user}]
        blocks.extend(frame_blocks(record))
        payload = json.dumps({"type": "user", "message": {"role": "user", "content": blocks},
                              "parent_tool_use_id": None}, ensure_ascii=False) + "\n"
    else:
        payload = user
    cmd = command(args, system, include_frames)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["ENABLE_TOOL_SEARCH"] = "false"
    attempts = getattr(args, "max_attempts", 3)
    for attempt in range(attempts):
        with _START_LOCK:
            interval = getattr(args, "request_interval", 0)
            if interval:
                time.sleep(max(0.0, interval - (time.monotonic() - _LAST_START)))
            _LAST_START = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="vh-claude-") as cwd:
                result = _invoke(cmd, payload, cwd, env,
                                 cache / f"{identity}.trace.jsonl",
                                 cache / f"{identity}.stderr.log", args.timeout)
            saved.write_text(json.dumps({"request_sha256": identity, "text": result["result"],
                "model": args.model, "usage": result.get("usage"),
                "model_usage": result.get("modelUsage"),
                "total_cost_usd": result.get("total_cost_usd"),
                "include_frames": include_frames}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result["result"]
        except (RuntimeError, TimeoutError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
