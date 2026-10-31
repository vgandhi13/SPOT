"""Native-tool EVA loop; six counted responses plus bounded tool-error recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from ..glimpse.api import APIRequestError, RetryPolicy, add_retry_arguments, read_key, request, retry_policy_from_args
from .data import load_tasks, sha256
from .eva_tools import EVAFrameTool
from .history import History, compact
from .prompts import FINAL_TURN_WARNING, FRAME_TOOL, build_messages, extract_answer

DEFAULT_TOOL_ERROR_RETRIES = 10
DEFAULT_VISIBLE_TOKEN_RESERVE = 700
DEFAULT_RESPONSE_TOKENS = 65536
TOOL_ERROR_FEEDBACK = (
    "All tool calls in your last response failed. Read the tool errors above. "
    "That response did not consume a turn. Correct the arguments and try again, "
    "or give your final option-label answer using the available evidence. "
    'Times must use quoted MM:SS with seconds from 00 to 59 (e.g. "1:00", not "0:60").'
)


def add_tool_error_arguments(parser):
    parser.add_argument("--max-tool-error-retries", type=int, default=DEFAULT_TOOL_ERROR_RETRIES,
                        help="Extra responses per sample after entirely failed tool batches (default: 10)")


def add_reasoning_arguments(parser):
    parser.add_argument("--visible-token-reserve", type=int, default=DEFAULT_VISIBLE_TOKEN_RESERVE,
                        help="Best-effort output space for text/tool calls, subtracted from reasoning budget; 0 disables (default: 700)")


def reasoning_request(response_tokens, visible_token_reserve):
    if response_tokens < 1 or not 0 <= visible_token_reserve < response_tokens:
        raise ValueError("visible-token-reserve must be nonnegative and smaller than response-tokens")
    settings = {"enabled": True, "exclude": False}
    if visible_token_reserve:
        settings["max_tokens"] = response_tokens - visible_token_reserve
    return settings


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def trajectory(task, tool, output, *, policy, model="elorian/glimpse", max_turns=6,
               response_tokens=DEFAULT_RESPONSE_TOKENS, temperature=0.0, send=None, key=None, retry_policy=None,
               max_tool_error_retries=DEFAULT_TOOL_ERROR_RETRIES,
               visible_token_reserve=DEFAULT_VISIBLE_TOKEN_RESERVE, initial_messages=None, tool_reminder=None):
    """send/tool are injectable: offline tests exercise the actual request loop."""
    if max_turns < 1 or response_tokens < 1:
        raise ValueError("max_turns and response_tokens must be positive")
    if max_tool_error_retries < 0:
        raise ValueError("max_tool_error_retries must be nonnegative")
    reasoning = reasoning_request(response_tokens, visible_token_reserve)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if initial_messages is None:
        messages, raw_question = build_messages(task["prompt"], tool.duration, tool.resolution, policy)
    else:
        # Local matched-model evaluations replay the exact saved source prompt.
        messages, raw_question = deepcopy(initial_messages), task["raw_question"]
    history = History(messages, policy, tool_reminder=tool_reminder)
    atomic_json(output / "prompt.json", compact(messages))
    atomic_json(output / "status.json", {"state": "running"})
    started = time.monotonic()
    usage, events = [], []
    retry_policy = retry_policy or RetryPolicy()
    attempts = []
    answer, stop, turn = "", "max_turns_exceeded", 0
    counted_turns, tool_error_retries = 0, 0
    final_warning_sent = False
    final_text = ""
    try:
        with (output / "trace.jsonl").open("w") as trace, \
                (output / "requests.jsonl").open("w") as requests, \
                (output / "api_attempts.jsonl").open("w") as api_log, \
                (output / "tools.jsonl").open("w") as tools_log, \
                (output / "notes.md").open("w") as notes:
            def emit(target, record):
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()

            while counted_turns < max_turns:
                # `turn` remains a unique API-response ID for logs/HTML. Recovery
                # requests reuse only the budget turn, never the response ID.
                turn += 1
                budget_turn = counted_turns + 1
                if budget_turn == max_turns and not final_warning_sent:
                    history.messages.append({'role': 'user', 'content': FINAL_TURN_WARNING})
                    final_warning_sent = True
                payload = {"model": model, "messages": history.payload_messages(),
                    "tools": [deepcopy(FRAME_TOOL)], "tool_choice": "auto",
                    "max_tokens": response_tokens,
                    "reasoning": deepcopy(reasoning),
                    "provider": {"require_parameters": True}, "stream": False}
                if temperature is not None:
                    payload["temperature"] = temperature
                emit(requests, {"turn": turn, "budget_turn": budget_turn, "payload": compact(payload)})
                def record_attempt(record):
                    record = {"turn": turn, "budget_turn": budget_turn, **record}
                    attempts.append(record)
                    emit(api_log, record)

                response = (send(payload, key) if send is not None else
                            request(payload, key, policy=retry_policy, on_attempt=record_attempt))
                choices = response.get("choices", [])
                if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                    raise RuntimeError("OpenRouter returned an incomplete response")
                choice = choices[0]
                raw = choice["message"]
                if raw.get("content") is not None and not isinstance(raw["content"], str):
                    raise RuntimeError("Unexpected assistant content type")
                assistant = history.assistant(raw)
                final_text = assistant.get("content") or ""
                usage.append(response.get("usage") or {})
                # Raw traces are immutable audit artifacts, distinct from replay.
                emit(trace, {"turn": turn, "budget_turn": budget_turn, "message": raw, "usage": usage[-1],
                             "finish_reason": choice.get("finish_reason"),
                             "id": response.get("id"), "model": response.get("model"),
                             "provider": response.get("provider")})
                if final_text:
                    notes.write(f"## Turn {turn}\n\n{final_text}\n\n")
                    notes.flush()
                answer = extract_answer(final_text)
                if answer:
                    counted_turns += 1
                    stop = "answer_found"
                    break
                calls = raw.get("tool_calls") or []
                if not calls:
                    counted_turns += 1
                    # Preserve EVA's stop on a response containing no action/answer.
                    stop = "no_answer_no_tool_call"
                    break
                if not isinstance(calls, list) or any(not isinstance(c, dict) or not c.get("id") for c in calls):
                    raise RuntimeError("Malformed native tool calls")
                ids = [c["id"] for c in calls]
                if len(set(ids)) != len(ids):
                    raise RuntimeError("Duplicate native tool-call IDs in a response")
                observations = []
                errors = []
                # Intentionally sequential even though the API can emit a batch.
                for c, call in enumerate(calls, 1):
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    event = {"call": len(events) + 1, "turn": turn, "budget_turn": budget_turn,
                             "tool_call_id": call["id"], "tool": function.get("name"),
                             "arguments": arguments}
                    try:
                        if function.get("name") != "frame_select":
                            raise ValueError("Unknown tool; only frame_select is available")
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be an object")
                        result = tool.execute(arguments)
                        event.update(status="ok", result=result)
                    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}", "frames": []}
                        event.update(status="error", result=result)
                        errors.append(result["error"])
                    events.append(event)
                    history.tool(call["id"], result, f"frames/turn_{turn}/call_{c}")
                    observations.append((call["id"], result))
                    emit(tools_log, event)
                # Resolve all native calls before sending images as user content,
                # as in claude_eval; every result is a separately evictable unit.
                for call_id, result in observations:
                    history.observation(call_id, result)
                history.reminder(raw_question)
                if len(errors) == len(calls):
                    if tool_error_retries >= max_tool_error_retries:
                        stop = "tool_error_retry_limit"
                        break
                    tool_error_retries += 1
                    history.messages.append({"role": "user", "content": TOOL_ERROR_FEEDBACK})
                    continue
                counted_turns += 1
        estimated = sum(e["result"].get("estimated_visual_tokens", 0) for e in events)
        totals = {name: sum(u.get(name, 0) or 0 for u in usage)
                  for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}
        gt = task.get("gt")
        # Match accuracy.py's whitespace/case comparison without normalizing the
        # recorded answer or accepting labels accompanied by descriptions.
        correct = bool(answer) and gt is not None and str(answer).strip().upper() == str(gt).strip().upper()
        result = {"index": task["index"], "video_id": task.get("video_id"),
            "question_id": task.get("question_id"), "gt": gt, "answer": answer,
            "correct": correct, "stop_reason": stop, "num_rounds": counted_turns,
            "model_responses": turn, "tool_error_retries": tool_error_retries,
            "tool_calls": len(events), "tool_errors": sum(e["status"] == "error" for e in events),
            "total_tokens": (usage[-1].get("total_tokens", 0) if usage else 0),
            "usage_total": totals, "estimated_visual_tokens": estimated,
            "final_text": final_text, "policy": policy,
            "elapsed_seconds": time.monotonic() - started}
        result.update(api_attempts=len(attempts), api_retries=sum(a['outcome'] == 'retry' for a in attempts),
            failed_attempt_usage_total={name: sum((a.get('usage') or {}).get(name, 0) or 0
                for a in attempts if a['outcome'] != 'succeeded')
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost')})
        atomic_json(output / "result.json", result)
        atomic_json(output / "history.json", compact(history.payload_messages()))
        atomic_json(output / "status.json", {"state": "completed", "stop_reason": stop})
        return result
    except BaseException as exc:
        atomic_json(output / "history.json", compact(history.payload_messages()))
        status = {"state": "failed", "error_type": type(exc).__name__}
        if isinstance(exc, APIRequestError):
            status.update(api_status=exc.status, api_attempts=exc.attempts,
                          retryable=exc.retryable, failed_turn=turn)
        atomic_json(output / "status.json", status)
        raise


def main(policy):
    if policy != "full_history":
        raise ValueError("Only full_history is supported")
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="Existing prepared Claude/Eulorian data bundle")
    inputs.add_argument("--task", type=Path, help="Single JSON with video_path, question, options, answer")
    inputs.add_argument("--eva-jsonl", type=Path, help="Original EVA-format dataset JSONL")
    inputs.add_argument("--holmes-json", type=Path, help="Official Video-Holmes split JSON (e.g. train_Video-Holmes.json)")
    parser.add_argument("--video-root", type=Path, help="Video root for --eva-jsonl/--holmes-json")
    parser.add_argument("--question-ids", type=Path, help="With --holmes-json: whitespace-separated Question IDs to run")
    parser.add_argument("--exclude-question-ids", type=Path,
                        help="With --holmes-json: Question IDs to skip, e.g. a dataset's solved_ids.txt (ignored if missing)")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Fresh run directory on active/scratch storage")
    parser.add_argument("--model", default="elorian/glimpse")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Concurrent samples, as in EVA; calls within a sample execute sequentially")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--response-tokens", type=int, default=DEFAULT_RESPONSE_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-v", "--max-visual-tokens", type=int, default=12000)
    parser.add_argument("--maxp", type=int, default=720)
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    add_tool_error_arguments(parser)
    add_reasoning_arguments(parser)
    add_retry_arguments(parser)
    args = parser.parse_args()
    try:
        retry_policy = retry_policy_from_args(args)
        reasoning = reasoning_request(args.response_tokens, args.visible_token_reserve)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.samples, args.max_concurrent, args.max_turns, args.response_tokens, args.max_visual_tokens, args.maxp) < 1 or args.sample_offset < 0:
        parser.error("Budgets/counts must be positive and sample-offset nonnegative")
    if args.max_tool_error_retries < 0:
        parser.error("max-tool-error-retries must be nonnegative")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real-video evaluation requires a Slurm allocation")
    output = args.output.resolve()
    if any(output.is_relative_to(root) for root in (Path("/home"), Path("/project"))):
        raise ValueError("Use active /work or an allocated scratch workspace for run media")
    key = read_key(os.environ["OPENROUTER_API_KEY_FILE"])
    tasks = load_tasks(args)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent.parent
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(policy=policy, tool_protocol="native_api", time_format="mmss",
        tool_error_turn_policy="all_failed_batches_do_not_count",
        api_retry_policy=asdict(retry_policy),
        final_turn_warning=FINAL_TURN_WARNING,
        replay_reasoning=True,
        retained_tool_results=None,
        reasoning_request=reasoning, visible_token_reserve_enforcement="provider_best_effort",
        api="OpenRouter", tool_count_cap=None, execution="sequential_within_each_response",
        slurm_job_id=os.environ["SLURM_JOB_ID"], python=sys.version,
        extractor_sha256=sha256(source / "shared/select_frame_fallback.py"),
        source_sha256={str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))})
    atomic_json(output / "config.json", config)
    atomic_json(output / "samples.json", [{k: v for k, v in t.items() if k != "prompt"} for t in tasks])
    atomic_json(output / "status.json", {"state": "running"})
    completed = []

    def run_one(pair):
        i, task = pair
        folder = output / f"sample_{i:04d}"
        try:
            tool = EVAFrameTool(task["video"], folder / "media", max_visual_tokens=args.max_visual_tokens,
                                maxp=args.maxp, fallback=args.fallback)
            return trajectory(task, tool, folder, policy=policy, model=args.model,
                              max_turns=args.max_turns, response_tokens=args.response_tokens,
                              temperature=args.temperature, key=key, retry_policy=retry_policy,
                              max_tool_error_retries=args.max_tool_error_retries,
                              visible_token_reserve=args.visible_token_reserve)
        except Exception as exc:
            # One failed sample (e.g. a context-length 400) must not abort the run;
            # its status.json records the failure and it has no result.json.
            return {"index": task["index"], "question_id": task.get("question_id"), "gt": task.get("gt"),
                    "answer": "", "correct": False, "tool_calls": None,
                    "stop_reason": f"sample_failed:{type(exc).__name__}", "error": str(exc)}

    try:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrent, len(tasks))) as pool:
            for result in pool.map(run_one, enumerate(tasks)):
                completed.append(result)
                atomic_json(output / "summary.json", {"completed": len(completed), "requested": len(tasks),
                    "correct": sum(r["correct"] for r in completed),
                    "accuracy": sum(r["correct"] for r in completed) / len(completed)})
                print(json.dumps({k: result[k] for k in ("index", "answer", "correct", "tool_calls", "stop_reason")}), flush=True)
        (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in completed))
        atomic_json(output / "status.json", {"state": "completed"})
    except BaseException as exc:
        atomic_json(output / "status.json", {"state": "failed", "error_type": type(exc).__name__})
        raise
