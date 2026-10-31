The memory prompts and memory-creation pipeline (`Memories/prompts.py`,
`Memories/pipeline.py`) and the model transports (`openrouter.py`, `claude_code.py`)
are adapted from the SPOTv2 `skill_creation` package, which adapts AIMING Lab's
SkillRL. See `LICENSE` for its MIT terms. The memory prompts are unchanged; the
pipeline reads SPOTv3's exported Glimpse trajectories (`eval/glimpse/export.py`).
