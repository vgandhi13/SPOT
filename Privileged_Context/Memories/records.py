"""Load an exported Glimpse run (eval/glimpse/export.py layout) as a memory source record."""
import hashlib
import json
from pathlib import Path
import re

from .prompts import CATEGORIES


def load_record(run_dir, holmes_rows):
    """Join run_<i>/trajectory.json and raw_trace.jsonl with the question's Video-Holmes row."""
    run_dir = Path(run_dir).resolve()
    raw = (run_dir / "trajectory.json").read_bytes()
    trajectory = json.loads(raw)
    row = holmes_rows[int(trajectory["question_id"])]
    # A few train labels carry annotation suffixes, e.g. "PAR-1" or "SR（附加题）"; keep the category.
    question_type = re.match(r"[A-Z]*", row["Question Type"]).group()
    if question_type not in CATEGORIES or row["Answer"] != trajectory["gt"]:
        raise ValueError(f"Trajectory does not match its Video-Holmes question: {run_dir}")
    trace = [json.loads(line) for line in (run_dir / "raw_trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    answer = trajectory["answer"] or ""
    tool_events = [{"turn": turn["turn"], "tool_call_id": call["tool_call_id"], "status": call["status"],
                    "arguments": call["arguments"],
                    "result": {"error": call["error"], "effective_arguments": call["effective_arguments"],
                               "frames": call["frames"]}}
                   for turn in trajectory["turns"] for call in turn["tool_calls"]]
    return {"question_id": row["Question ID"], "video_id": row["video ID"],
            "Question Type": question_type, "question": row["Question"], "options": row["Options"],
            "ground_truth": row["Answer"], "answer": answer, "correct": trajectory["correct"],
            "answer_format_valid": answer in row["Options"],
            "outcome": "Success" if trajectory["correct"] else "Failure",
            "source_split": "train",
            "trace": [{"turn": r["turn"], "message": r["message"], "finish_reason": r.get("finish_reason")} for r in trace],
            "tool_events": tool_events,
            "run_dir": str(run_dir), "trajectory_sha256": hashlib.sha256(raw).hexdigest()}


def holmes_index(path):
    return {r["Question ID"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))}
