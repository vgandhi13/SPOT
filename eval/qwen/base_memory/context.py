"""Inject this question's successful memory or failure-specific mistakes."""
from copy import deepcopy
import json
from pathlib import Path


def memory_prefix(task, memory):
    """Use only this question's memory, or its failure-specific warnings."""
    from skill_creation.trajectory_insights import GUIDANCE, PROSE_PLAN_POLICIES, VARIANT, validate_memory, variant_of
    if variant_of(memory) == VARIANT:
        validate_memory(memory)
        if (memory['origin_question_id'] != task['question_id'] or
                memory['memory_id'] != f"mem_vh_q{task['question_id']}" or
                memory['tags']['question_type'] != task['question_type']):
            raise ValueError('Insights do not match the evaluation question')
        label = 'Question-Specific Trajectory Insights'
        if memory['input_policy'] in PROSE_PLAN_POLICIES:
            content = memory['content']
            paragraphs = [f"{i}. {step['instruction'].strip()}"
                          for i, step in enumerate(content['plan_steps'], 1)]
            if content['reasoning_cautions']:
                paragraphs.append('Reasoning cautions:\n' + '\n'.join(
                    f'- {caution.strip()}' for caution in content['reasoning_cautions']))
            guidance = '\n\n'.join(paragraphs)
        else:
            guidance = json.dumps(memory['content'], ensure_ascii=False, indent=2)
        return (f'[{label}]\n{GUIDANCE}\n' + guidance
                + f'\n[/{label}]\n\n')
    if (memory["origin_question_id"] != task["question_id"] or
            memory["memory_id"] != f"mem_vh_q{task['question_id']}" or
            memory["tags"]["question_type"] != task["question_type"] or
            memory["content"]["task_meta"]["question_type"] != task["question_type"]):
        raise ValueError("Memory does not match the evaluation question")
    outcome = memory["tags"]["outcome"]
    if outcome == "Success":
        payload = {"contextual_description": memory["contextual_description"], "content": memory["content"]}
        label = "Question-Specific Memory"
        guidance = "The following saved memory comes from a successful prior attempt on this same question."
    elif outcome == "Failure":
        payload = {"mistakes_to_avoid": memory["content"]["strategic_guidelines"]["mistakes_to_avoid"]}
        if not isinstance(payload["mistakes_to_avoid"], list):
            raise ValueError("Expected a list of question-specific mistakes")
        label = "Question-Specific Mistakes to Avoid"
        guidance = "Avoid the following mistakes from a failed prior attempt on this same question."
    else:
        raise ValueError("Unknown source memory outcome")
    return (f"[Privileged {label}]\n{guidance}\n" +
            json.dumps(payload, ensure_ascii=False, indent=2) + f"\n[/Privileged {label}]\n\n")


def load_question_memories(path, tasks):
    from skill_creation.pipeline import validate_memories
    from skill_creation.trajectory_insights import VARIANT, variant_of
    memories = json.loads(Path(path).read_text())
    if not isinstance(memories, list) or not memories:
        raise ValueError("Expected a nonempty memory array")
    variants = {variant_of(m) for m in memories}
    if len(variants) != 1:
        raise ValueError('One collection cannot mix memory variants')
    if VARIANT in variants:
        validate_memories(memories)
    indexed = {m["origin_question_id"]: m for m in memories}
    if len(indexed) != len(memories) or set(indexed) != {t["question_id"] for t in tasks}:
        raise ValueError("Require exactly one memory for every evaluation question")
    for task in tasks:
        memory_prefix(task, indexed[task["question_id"]])
    return indexed


def build_messages(task, memories):
    if memories is None:
        raise ValueError("Memory condition requires question-specific memories")
    messages = deepcopy(task["initial_messages"])
    messages[0]["content"] = memory_prefix(task, memories[task["question_id"]]) + messages[0]["content"]
    return messages
