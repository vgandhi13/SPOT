"""Privileged (SkillRL-style) memory creation and validation, adapted from SPOTv2 skill_creation.

Each memory sees the full trajectory, the correct label and the recorded outcome.
Text-only by default; include_frames attaches the frames the agent actually received.
"""
import json
from pathlib import Path

from ..transport import call_model
from ..util import write
from .prompts import CATEGORIES, EVIDENCE_RULES, FRAME_EVIDENCE_RULES, MEMORY_PROMPTS

TEXT_ONLY = "text_only_agent_reported"
WITH_FRAMES = "sampled_frames_and_agent_reports"


def memory_id(record):
    return f"mem_vh_q{record['question_id']}"


def reasoning_text(message):
    texts = []
    for v in (message.get("reasoning"), message.get("reasoning_content")):
        if isinstance(v, str) and v and v not in texts:
            texts.append(v)
    for detail in message.get("reasoning_details") or []:
        v = detail.get("text")
        if isinstance(v, str) and v and v not in texts:
            texts.append(v)
    return "\n\n".join(texts)


def trajectory_context(record, include_frames=False):
    turns = []
    for row in record["trace"]:
        m = row["message"]
        tools = []
        for event in record["tool_events"]:
            if event["turn"] != row["turn"]:
                continue
            result = event["result"]
            tools.append({"call_id": event["tool_call_id"], "status": event["status"],
                          "arguments": event["arguments"], "error": result.get("error"),
                          "effective_arguments": result.get("effective_arguments"),
                          "sampled_frame_timestamps": [f["timestamp"] for f in result.get("frames", [])],
                          "visual_contents": ("Provided as labeled images after this trajectory JSON."
                                              if include_frames else "Not provided; only the agent's later text reports are available.")})
        turns.append({"turn": row["turn"], "agent_reported_reasoning": reasoning_text(m),
                      "assistant_content": m.get("content"), "tool_calls": m.get("tool_calls", []),
                      "tool_results": tools, "finish_reason": row.get("finish_reason")})
    return {"question": record["question"], "options": record["options"],
            "question_type": record["Question Type"], "category_definition": CATEGORIES[record["Question Type"]],
            "recorded_outcome": record["outcome"], "agent_answer": record["answer"],
            "answer_format_valid": record["answer_format_valid"],
            "correct_label": record["ground_truth"],
            "evidence_mode": WITH_FRAMES if include_frames else TEXT_ONLY,
            "turns": turns}


def memory_jobs(record, include_frames=False):
    context = json.dumps(trajectory_context(record, include_frames), ensure_ascii=False)
    parts = ["contextual_description"]
    if record["outcome"] == "Success":
        parts.append("refined_trajectory")
    parts.append("strategic_guidelines")
    return [{"id": f"{memory_id(record)}_{part}", "memory_id": memory_id(record), "part": part,
             "messages": [{"role": "system", "content": (MEMORY_PROMPTS[part].replace(EVIDENCE_RULES, FRAME_EVIDENCE_RULES)
                                                if include_frames else MEMORY_PROMPTS[part])},
                          {"role": "user", "content": context}]} for part in parts]


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def require_strings(obj, keys):
    if not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in keys):
        raise ValueError(f"Require nonempty string fields: {keys}")


def make_memory(record, parts, evidence_mode=TEXT_ONLY):
    desc = parts["contextual_description"].strip()
    if not desc:
        raise ValueError("Empty description")
    refined = None
    if record["outcome"] == "Success":
        refined = parse_json_response(parts["refined_trajectory"])
        if isinstance(refined, dict):
            refined = refined.get("refined_trajectory")
        if not isinstance(refined, list):
            raise ValueError("Refined trajectory must be an array")
        turns = {row["turn"] for row in record["trace"]}
        indices = []
        for step in refined:
            require_strings(step, ("action", "critical_observation", "reasoning"))
            if type(step.get("step_index")) is not int or step["step_index"] not in turns:
                raise ValueError("Refined step must reference an existing turn")
            indices.append(step["step_index"])
        if indices != sorted(set(indices)):
            raise ValueError("Refined steps must be unique and chronological")
    strategic = parse_json_response(parts["strategic_guidelines"])
    if isinstance(strategic, dict) and "strategic_guidelines" in strategic:
        strategic = strategic["strategic_guidelines"]
    if not isinstance(strategic, dict) or not isinstance(strategic.get("mistakes_to_avoid"), list):
        raise ValueError("Invalid strategic guidelines")
    if record["outcome"] == "Success":
        require_strings(strategic, ("planning_pattern",))
        if strategic["mistakes_to_avoid"]:
            raise ValueError("Successful memory must have an empty mistakes list")
    elif strategic.get("planning_pattern", "missing") is not None:
        raise ValueError("Failed memory must have a null planning pattern")
    for item in strategic["mistakes_to_avoid"]:
        require_strings(item, ("trigger_condition", "bad_action"))
    return {"memory_id": memory_id(record), "contextual_description": desc,
            "tags": {"environment": "VideoHolmes", "question_type": record["Question Type"],
                     "outcome": record["outcome"], "source_split": record["source_split"]},
            "content": {"task_meta": {"original_goal": record["question"], "question_type": record["Question Type"]},
                        "refined_trajectory": refined, "strategic_guidelines": strategic},
            "origin_question_id": record["question_id"], "source_trajectory_sha256": record["trajectory_sha256"],
            "evidence_mode": evidence_mode}


class QuotaExhausted(RuntimeError):
    """The provider keeps rate-limiting (e.g. a daily free-model cap); stop and resume later."""


def memory_name(record):
    """q<ID>_run_<i> for an export.py run folder; one memory file per source run."""
    run = Path(record["run_dir"])
    return f"{run.parent.name}_{run.name}"


def generate_memories(records, args):
    """One memory file per record in output/memories/, skipping ones already made.

    A part failing validation is re-requested up to twice with feedback. Other per-record
    failures go to failures.jsonl and the run moves on; persistent HTTP 429 stops the run.
    """
    output = Path(args.output)
    folder = output / "memories"
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for r in records:
        path = folder / f"{memory_name(r)}.json"
        if path.exists():
            continue
        try:
            memory = generate_memory(r, args)
        except RuntimeError as exc:
            if "HTTP 429" in str(exc):
                raise QuotaExhausted(f"Rate limited at {memory_name(r)}; resume later") from None
            failure = f"{type(exc).__name__}: {exc}"
        except (ValueError, TypeError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        else:
            write(path, memory)
            done += 1
            print(f"{memory_name(r)}: {r['outcome']}", flush=True)
            continue
        with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"memory": memory_name(r), "error": failure}) + "\n")
        print(f"{memory_name(r)}: FAILED {failure}", flush=True)
    memories = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    write(output / "generated_memories.json", memories)
    return {"count": len(memories), "new": done, "path": str(output / "generated_memories.json")}


def generate_memory(r, args):
    output = Path(args.output)
    include_frames = getattr(args, "include_frames", False)
    evidence_mode = WITH_FRAMES if include_frames else TEXT_ONLY
    parts = {job["part"]: call_model(job["messages"], args, output / "responses", record=r)
             for job in memory_jobs(r, include_frames)}
    for validation_attempt in range(3):
        try:
            memory = make_memory(r, parts, evidence_mode)
            break
        except (ValueError, TypeError, KeyError) as exc:
            if validation_attempt == 2:
                raise
            detail = str(exc)
            part = "strategic_guidelines"
            if any(word in detail.lower() for word in ("refined", "step", "chronological", "critical_observation")):
                part = "refined_trajectory"
            elif "description" in detail.lower():
                part = "contextual_description"
            else:
                for candidate in ("refined_trajectory", "strategic_guidelines"):
                    if candidate in parts:
                        try:
                            parse_json_response(parts[candidate])
                        except (ValueError, TypeError):
                            part = candidate
                            break
            job = next(job for job in memory_jobs(r, include_frames) if job["part"] == part)
            feedback = (f"The previous {part} response failed validation: {detail}.\n"
                f"The recorded outcome is {r['outcome']}. Existing source turn numbers are "
                f"{[row['turn'] for row in r['trace']]}. For refined trajectories, use only these source turn "
                "numbers, once each, in increasing chronological order. Combine relevant actions within "
                "one source turn; do not number newly invented logical substeps.\n"
                "Follow the original abstraction instructions: replace specific people/events with "
                "placeholders, qualify agent-reported observations, and do not reproduce the answer key. "
                f"Return only the corrected {part}, in the format requested by the system prompt.")
            messages = job["messages"] + [{"role": "assistant", "content": parts[part]},
                {"role": "user", "content": feedback}]
            with (output / "validation_retries.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"memory_id": memory_id(r), "part": part,
                    "attempt": validation_attempt + 1, "validation_error": detail,
                    "messages": messages}, ensure_ascii=False) + "\n")
            print(f"{memory_id(r)}: retrying {part} after validation failure", flush=True)
            parts[part] = call_model(messages, args, output / "responses", record=r)
    return memory


def validate_memories(memories):
    if not isinstance(memories, list) or not memories:
        raise ValueError("Expected a nonempty memory array")
    ids = set()
    for m in memories:
        if m["memory_id"] in ids:
            raise ValueError("Duplicate memory ID")
        ids.add(m["memory_id"])
        if m["tags"]["question_type"] not in CATEGORIES or m["tags"]["outcome"] not in ("Success", "Failure"):
            raise ValueError("Invalid category/outcome metadata")
        if m.get("evidence_mode") not in (TEXT_ONLY, WITH_FRAMES):
            raise ValueError("Memory evidence mode does not match this pipeline")
        s = m["content"]["strategic_guidelines"]
        if not isinstance(s["mistakes_to_avoid"], list):
            raise ValueError("Invalid mistakes list")
        if m["tags"]["outcome"] == "Success":
            require_strings(s, ("planning_pattern",))
            if s["mistakes_to_avoid"]:
                raise ValueError("Successful memory must have no mistakes")
        elif s["planning_pattern"] is not None:
            raise ValueError("Failed memory must have null planning pattern")
        for item in s["mistakes_to_avoid"]:
            require_strings(item, ("trigger_condition", "bad_action"))
