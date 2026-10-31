"""Export a Glimpse run directory into per-question trajectory datasets, split by correctness.

For a prefix such as data/glimpse:
  <prefix>_correct/q<ID>/run_<i>/     runs whose final answer matched the label
  <prefix>_incorrect/q<ID>/run_<i>/   all other completed runs
      trajectory.json   human-readable trajectory
      raw_trace.jsonl   unmodified API responses
      frames/turn_<t>/call_<c>/frame_<k>.png
  <prefix>_index.json, <prefix>_solved_ids.txt
run_<i> is the question's i-th exported attempt, numbered across both folders, so a
question solved on a later attempt appears in both. Each source sample is exported
once; re-running the export is a no-op for it. Samples that failed (no result.json)
are not exported.
Retry unsolved questions with run.py --exclude-question-ids <prefix>_solved_ids.txt.
"""
import argparse
import json
from pathlib import Path
import shutil


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def split_dirs(prefix):
    return {True: prefix.with_name(prefix.name + "_correct"),
            False: prefix.with_name(prefix.name + "_incorrect")}


def trajectories(prefix):
    return sorted(p for d in split_dirs(prefix).values() for p in d.glob("q*/run_*/trajectory.json"))


def exported_sources(prefix):
    """(run_dir, sample) pairs already in the dataset, to keep the export idempotent."""
    seen = set()
    for path in trajectories(prefix):
        source = json.loads(path.read_text())["source"]
        seen.add((source["run_dir"], source["sample"]))
    return seen


def next_run_name(prefix, question):
    taken = [int(p.name.split("_")[1]) for d in split_dirs(prefix).values()
             for p in (d / question).glob("run_*")]
    return f"run_{max(taken, default=0) + 1}"


def export_sample(sample, task, config, run_dir, prefix):
    result = json.loads((sample / "result.json").read_text())
    prompt = json.loads((sample / "prompt.json").read_text())
    trace, tools = lines(sample / "trace.jsonl"), lines(sample / "tools.jsonl")
    question = f"q{int(result['question_id']):04d}"
    target = split_dirs(prefix)[bool(result["correct"])] / question / next_run_name(prefix, question)
    target.mkdir(parents=True)
    turns = []
    for response in trace:
        message = response["message"]
        calls = []
        for c, event in enumerate((e for e in tools if e["turn"] == response["turn"]), 1):
            frames = []
            for k, frame in enumerate(event["result"].get("frames", [])):
                rel = Path("frames") / f"turn_{response['turn']}" / f"call_{c}" / f"frame_{k:02d}.png"
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame["path"], target / rel)
                frames.append({"file": str(rel), "timestamp": mmss(frame["timestamp_seconds"])})
            calls.append({"tool_call_id": event["tool_call_id"], "arguments": event["arguments"],
                          "status": event["status"], "error": event["result"].get("error"),
                          "effective_arguments": event["result"].get("effective_arguments"),
                          "frames": frames})
        turns.append({"turn": response["turn"], "budget_turn": response["budget_turn"],
                      "reasoning": message.get("reasoning"), "content": message.get("content"),
                      "tool_calls": calls, "finish_reason": response.get("finish_reason"),
                      "usage": response.get("usage")})
    trajectory = {
        "question_id": result["question_id"], "video_id": result["video_id"],
        "question_type": task.get("question_type"), "prompt": prompt,
        "gt": result["gt"], "answer": result["answer"], "correct": result["correct"],
        "stop_reason": result["stop_reason"], "num_rounds": result["num_rounds"],
        "tool_calls": result["tool_calls"], "tool_errors": result["tool_errors"],
        "usage_total": result["usage_total"], "elapsed_seconds": result["elapsed_seconds"],
        "model": config["model"], "temperature": config["temperature"],
        "max_turns": config["max_turns"], "max_visual_tokens": config["max_visual_tokens"],
        "source": {"run_dir": str(run_dir), "sample": sample.name,
                   "slurm_job_id": config.get("slurm_job_id")},
        "turns": turns}
    (target / "trajectory.json").write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(sample / "trace.jsonl", target / "raw_trace.jsonl")
    return target


def write_index(prefix):
    index = {}
    for path in trajectories(prefix):
        t = json.loads(path.read_text())
        entry = index.setdefault(str(t["question_id"]),
                                 {"runs": 0, "correct_runs": [], "incorrect_runs": [], "cost": 0.0})
        entry["runs"] += 1
        entry["cost"] += t["usage_total"].get("cost") or 0
        entry["correct_runs" if t["correct"] else "incorrect_runs"].append(path.parent.name)
    index = dict(sorted(index.items(), key=lambda item: int(item[0])))
    for entry in index.values():
        for key in ("correct_runs", "incorrect_runs"):
            entry[key].sort(key=lambda name: int(name.split("_")[1]))
    solved = sum(bool(e["correct_runs"]) for e in index.values())
    summary = {"questions": len(index), "solved": solved,
               "runs": sum(e["runs"] for e in index.values()),
               "correct_runs": sum(len(e["correct_runs"]) for e in index.values()),
               "cost": round(sum(e["cost"] for e in index.values()), 4)}
    prefix.with_name(prefix.name + "_index.json").write_text(
        json.dumps({"summary": summary, "questions": index}, indent=2) + "\n")
    solved_ids = [int(q) for q, e in index.items() if e["correct_runs"]]
    prefix.with_name(prefix.name + "_solved_ids.txt").write_text("".join(f"{q}\n" for q in solved_ids))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory written by eval/glimpse/run.py")
    parser.add_argument("prefix", type=Path, help="Dataset prefix, e.g. data/glimpse")
    args = parser.parse_args()
    run_dir, prefix = args.run_dir.resolve(), args.prefix.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    tasks = json.loads((run_dir / "samples.json").read_text())
    prefix.parent.mkdir(parents=True, exist_ok=True)
    seen = exported_sources(prefix)
    exported, skipped = 0, []
    for i, task in enumerate(tasks):
        sample = run_dir / f"sample_{i:04d}"
        if (str(run_dir), sample.name) in seen:
            continue
        if not (sample / "result.json").is_file():
            # API failures/incomplete samples stay unexported and are retried later.
            skipped.append(task.get("question_id"))
            continue
        export_sample(sample, task, config, run_dir, prefix)
        exported += 1
    summary = write_index(prefix)
    print(json.dumps({"exported": exported, "skipped_incomplete": skipped, **summary}))


if __name__ == "__main__":
    main()
