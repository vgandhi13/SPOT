"""Prepend general skills, category skills, and common mistakes."""
from copy import deepcopy
from skill_creation.export_sdar import skill_prefix


def build_messages(task, skills_dir):
    if skills_dir is None:
        raise ValueError("Skills condition requires exported Markdown")
    messages = deepcopy(task["initial_messages"])
    messages[0]["content"] = skill_prefix(skills_dir, task["question_type"]) + messages[0]["content"]
    return messages
