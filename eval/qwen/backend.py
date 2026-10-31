"""Matched local Qwen3-VL evaluation using the saved full-history Glimpse prompts."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib import error, request

from ..shared.data import sha256
from ..shared.eva_tools import EVAFrameTool
from ..shared.history import History, compact
from ..shared.prompts import FRAME_TOOL
from ..shared.runner import atomic_json, trajectory

from .prompts import QWEN_TOOL_REMINDER, initial_messages, load_question_memories, memory_prefix

MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def server_environment(environment, ipc_directory):
    """Keep vLLM and Python multiprocessing sockets below Linux's path limit."""
    directory = str(Path(ipc_directory).resolve())
    if len(os.fsencode(directory + "/" + "0" * 36)) > 103:
        raise ValueError("Use a shorter node-local IPC directory")
    return {**environment, "TMPDIR": directory, "VLLM_RPC_BASE_PATH": directory}


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    tasks = data["tasks"]
    ids = [t["question_id"] for t in tasks]
    if not tasks or len(set(ids)) != len(ids):
        raise ValueError("Require nonempty unique evaluation questions")
    for t in tasks:
        if t["question_type"] not in ("SR", "PAR", "MHR", "IMC", "TCI", "TA", "CTI"):
            raise ValueError("Unknown labeled question category")
        if not isinstance(t["initial_messages"], list) or t["initial_messages"][0]["role"] != "system":
            raise ValueError("Missing exact saved system prompt")
        if t["gt"] not in t["options"] or not Path(t["video"]).is_file():
            raise ValueError("Invalid task label or missing video")
    return data


class LocalAPI:
    def __init__(self, endpoint, key):
        self.endpoint, self.key = endpoint, key

    def send(self, payload, key=None, log_path=None):
        # OpenRouter provider/reasoning knobs do not apply to this Instruct model.
        allowed = ("model", "messages", "tools", "tool_choice", "max_tokens", "temperature", "stream")
        actual = {k: payload[k] for k in allowed if k in payload}
        actual["seed"] = 0
        if log_path:
            with Path(log_path).open("a") as stream:
                stream.write(json.dumps(compact(actual), ensure_ascii=False)+"\n")
        req = request.Request(self.endpoint, data=json.dumps(actual).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer "+self.key})
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=900) as response:
                    return json.load(response)
            except error.HTTPError as exc:
                detail = exc.read().decode(errors="replace").replace(self.key, "[redacted]")[:1000]
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Local inference HTTP {exc.code}: {detail}") from None
            except (error.URLError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError("Local inference transport failed") from None
            time.sleep(2 ** attempt)


def smoke(api, task, root):
    """Check native tool parsing and real frame/image handling before the suite."""
    root.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "system", "content": "Use the native frame_select tool when asked."},
                {"role": "user", "content": 'Call frame_select now with start_time="0:00", end_time="0:01", nframes=1.'}]
    payload = {"model": MODEL, "messages": messages, "tools": [FRAME_TOOL], "tool_choice": "auto",
               "max_tokens": 256, "temperature": 0.0, "stream": False}
    response = api.send(payload, log_path=root / "requests.jsonl")
    atomic_json(root / "tool_response.json", response)
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "frame_select":
        raise RuntimeError("Native frame_select tool smoke check failed")
    tool = EVAFrameTool(task["video"], root / "media")
    result = tool.execute({"start_time": "0:00", "end_time": "0:01", "nframes": 1})
    history = History([{"role": "system", "content": "Briefly describe the supplied video frame."}], "full_history")
    history.observation("smoke_frame", result)
    image_payload = {"model": MODEL, "messages": history.payload_messages(), "max_tokens": 256,
                     "temperature": 0.0, "stream": False}
    image_response = api.send(image_payload, log_path=root / "requests.jsonl")
    atomic_json(root / "image_response.json", image_response)
    if not (image_response["choices"][0]["message"].get("content") or "").strip():
        raise RuntimeError("Real-image smoke check produced no text")
    atomic_json(root / "status.json", {"state": "passed", "native_tool_calls": len(calls), "real_frames": len(result["frames"])})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--condition", choices=("plain", "skills", "memory"), required=True)
    cli.add_argument("--skills-dir", type=Path)
    cli.add_argument("--memories", type=Path)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--context-tokens", type=int, default=262144)
    cli.add_argument("--tensor-parallel-size", type=int, default=1)
    cli.add_argument("--response-tokens", type=int, default=32768)
    cli.add_argument("--max-turns", type=int, default=6)
    cli.add_argument("--temperature", type=float, default=0.5)
    cli.add_argument("--smoke-only", action="store_true")
    args = cli.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        cli.error("--temperature must be finite and nonnegative")
    if args.tensor_parallel_size < 1:
        cli.error("--tensor-parallel-size must be positive")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Local model/video evaluation requires an allocated Slurm GPU")
    if args.output.resolve().is_relative_to(Path("/project")) or args.output.resolve().is_relative_to(Path("/home")):
        raise ValueError("Model/video outputs must use active work or allocated scratch storage")
    if args.context_tokens <= args.response_tokens or min(args.response_tokens, args.max_turns) < 1:
        raise ValueError("Invalid context/response/turn budgets")
    data = load_manifest(args.manifest)
    tasks = data["tasks"]
    memories = load_question_memories(args.memories, tasks) if args.condition == "memory" and args.memories else None
    # Fail missing/invalid context before spending time loading weights.
    for task in tasks:
        initial_messages(task, args.condition, args.skills_dir, memories)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model=MODEL, model_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
                  manifest_sha256=sha256(args.manifest), task_count=len(tasks), policy="full_history",
                  prompt_source="exact saved initial prompts of the 48 source Glimpse trajectories",
                  tool_protocol="native_api", tool_parser="hermes", time_format="MM:SS", temperature=args.temperature, seed=0,
                  tool_count_cap=None, max_visual_tokens=12000, maxp=720, max_tool_error_retries=10,
                  tool_reminder=QWEN_TOOL_REMINDER,
                  skill_injection="SDAR wrapper prepended inside original system message" if args.condition == "skills" else None,
                  evaluation_scope="same questions used for skill memory extraction; not held-out",
                  slurm_job_id=os.environ["SLURM_JOB_ID"], python=sys.version,
                  versions={n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers")})
    if args.skills_dir:
        config["skill_export_sha256"] = sha256(args.skills_dir / "export_manifest.json")
    if memories is not None:
        from skill_creation.trajectory_insights import VARIANT, variant_of
        memory_variant = variant_of(next(iter(memories.values())))
        config.update(memories_sha256=sha256(args.memories),
                      memory_injection="successful source: complete contextual_description and content; failed source: mistakes_to_avoid only; prepended inside original system message",
                      memory_source_outcomes={o: sum(m["tags"].get("outcome") == o for m in memories.values())
                                              for o in ("Success", "Failure")})
        if memory_variant == VARIANT:
            config.update(memory_variant=memory_variant,
                          memory_injection="outcome-blind inspection suggestions; verify evidence and derive answer",
                          memory_source_outcomes=None)
        atomic_json(args.output / "memory_contexts.json", [
            {"question_id": t["question_id"], "memory_id": memories[t["question_id"]]["memory_id"],
             "source_outcome": memories[t["question_id"]]["tags"].get("outcome"),
             "prefix": memory_prefix(t, memories[t["question_id"]])} for t in tasks])
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "status.json", {"state": "starting", "requested": len(tasks)})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    base = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model_dir),
           "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--tensor-parallel-size", str(args.tensor_parallel_size), "--max-model-len", str(args.context_tokens),
           "--gpu-memory-utilization", "0.85", "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
           "--enable-chunked-prefill", "--enforce-eager", "--swap-space", "0", "--generation-config", "vllm",
           "--seed", "0", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
           "--limit-mm-per-prompt", '{"image":512,"video":0}', "--mm-processor-cache-gb", "0", "--skip-mm-profiling"]
    atomic_json(args.output / "server_command.json", cmd)
    server = None
    ipc_directory = None
    def interrupted(signum, frame):
        raise RuntimeError(f"Job interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        ipc_directory = tempfile.TemporaryDirectory(
            prefix=f"vh-{os.environ['SLURM_JOB_ID']}-", dir="/dev/shm")
        server_env = server_environment(
            {**os.environ, "VLLM_API_KEY": key}, ipc_directory.name)
        atomic_json(args.output / "runtime_paths.json", {
            "server_tmpdir": server_env["TMPDIR"],
            "vllm_rpc_base_path": server_env["VLLM_RPC_BASE_PATH"]})
        with (args.output / "vllm.log").open("a") as log:
            server = subprocess.Popen(cmd, env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {server.returncode}; inspect vllm.log")
                try:
                    with request.urlopen(request.Request(base+"/health", headers={"Authorization": "Bearer "+key}), timeout=2):
                        break
                except (error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Local model startup exceeded 20 minutes")
                    time.sleep(3)
            api = LocalAPI(base+"/v1/chat/completions", key)
            smoke(api, tasks[0], args.output / "smoke")
            if args.smoke_only:
                atomic_json(args.output / "status.json", {"state": "smoke_passed"})
                return
            atomic_json(args.output / "status.json", {"state": "running", "requested": len(tasks)})
            rows = []
            for task in tasks:
                folder = args.output / f"question_{task['question_id']}"
                try:
                    tool = EVAFrameTool(task["video"], folder / "media", max_visual_tokens=12000, maxp=720)
                    row = trajectory(task, tool, folder, policy="full_history", model=MODEL,
                        max_turns=args.max_turns, response_tokens=args.response_tokens, temperature=args.temperature,
                        max_tool_error_retries=10, visible_token_reserve=0,
                        initial_messages=initial_messages(task, args.condition, args.skills_dir, memories),
                        tool_reminder=QWEN_TOOL_REMINDER,
                        send=lambda payload, unused_key: api.send(payload, log_path=folder / "local_requests.jsonl"))
                    row["outcome"] = "correct" if row["correct"] else "wrong"
                    row["question_type"] = task["question_type"]
                except Exception as exc:
                    row = {"question_id": task["question_id"], "question_type": task["question_type"],
                           "correct": False, "outcome": "execution_error", "error": str(exc)[:1500]}
                    folder.mkdir(parents=True, exist_ok=True)
                    atomic_json(folder / "execution_error.json", row)
                rows.append(row)
                correct = sum(bool(r["correct"]) for r in rows)
                summary = {"requested": len(tasks), "completed": len(rows), "correct": correct,
                           "execution_errors": sum(r["outcome"] == "execution_error" for r in rows),
                           "accuracy_so_far": correct/len(rows), "accuracy_requested": correct/len(tasks),
                           "categories": {c: {"completed": sum(r["question_type"] == c for r in rows),
                                             "correct": sum(r["question_type"] == c and r["correct"] for r in rows)}
                                          for c in sorted({t["question_type"] for t in tasks})}}
                atomic_json(args.output / "summary.json", summary)
                atomic_json(args.output / "results.json", rows)
                print(json.dumps({k: row[k] for k in ("question_id", "outcome", "correct")}), flush=True)
            state = "completed_with_errors" if summary["execution_errors"] else "completed"
            atomic_json(args.output / "status.json", {"state": state, **summary})
            if summary["execution_errors"]:
                raise RuntimeError(f"{summary['execution_errors']} evaluation trajectories failed; all questions were attempted")
    except BaseException as exc:
        failure = {"state": "failed", "error": str(exc)[:1500]}
        atomic_json(args.output / "failure.json", failure)
        existing = json.loads((args.output / "status.json").read_text())
        if existing.get("state") != "completed_with_errors":
            atomic_json(args.output / "status.json", failure)
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if ipc_directory is not None:
            ipc_directory.cleanup()


if __name__ == "__main__":
    main()
