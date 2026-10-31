"""Dispatch matched Qwen prompt construction to the selected variant."""
from .base.context import build_messages as base_messages
from .base_memory.context import build_messages as memory_messages, load_question_memories, memory_prefix
from .base_skills_mistakes.context import build_messages as skills_messages

QWEN_TOOL_REMINDER = (
    "If more information is needed, reason briefly in your assistant text about "
    "what remains uncertain and why the next inspection will help, then call "
    "the frame selection tool again."
)


def initial_messages(task, condition, skills_dir=None, memories=None):
    if condition == "plain":
        return base_messages(task)
    if condition == "skills":
        return skills_messages(task, skills_dir)
    if condition == "memory":
        return memory_messages(task, memories)
    raise ValueError("Unknown evaluation condition")
