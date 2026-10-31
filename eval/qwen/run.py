"""Replay original full-history prompts with base, skills+mistakes, or own memory."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from storage import CONDITIONS, digest, read, run_directory, selected_tasks, validate_memories, write
from eval.qwen.prompts import QWEN_TOOL_REMINDER, initial_messages, load_question_memories
from skill_creation.trajectory_insights import VARIANT, VARIANTS, check_variant


def prepare(run, conditions, question_ids=None, skills_dir=None, memories_path=None, memory_variant='privileged'):
    check_variant(memory_variant)
    manifest = selected_tasks(run, question_ids)
    skills_dir = Path(skills_dir or run / 'skills')
    memories_path = Path(memories_path or run / 'memories/generated_memories.json')
    # Validate against complete source collection, then subset for native evaluator.
    memories = validate_memories(run, memories_path, memory_variant) if 'base_memory' in conditions else None
    if 'base_skills_mistakes' in conditions:
        exported_variant = read(skills_dir / 'export_manifest.json').get('memory_variant', 'privileged')
        if exported_variant != memory_variant:
            raise ValueError('Skill export does not match requested --memory-variant')
    selected_ids = {t['question_id'] for t in manifest['tasks']}
    indexed = {m['origin_question_id']: m for m in memories or [] if m['origin_question_id'] in selected_ids}
    for condition in conditions:
        for task in manifest['tasks']:
            initial_messages(task, CONDITIONS[condition], skills_dir, indexed)
    return manifest, skills_dir, memories_path, indexed


def main(condition=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    cli.add_argument('--conditions', nargs='+', choices=[condition] if condition else CONDITIONS, default=[condition] if condition else list(CONDITIONS))
    cli.add_argument('--question-ids', nargs='+', type=int)
    cli.add_argument('--skills-dir', type=Path)
    cli.add_argument('--memories', type=Path)
    cli.add_argument('--memory-variant', choices=VARIANTS, default='privileged')
    cli.add_argument('--model-dir', type=Path)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--temperature', type=float, default=0.5)
    cli.add_argument('--tensor-parallel-size', type=int, default=1)
    cli.add_argument('--context-tokens', type=int, default=262144)
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        cli.error('--temperature must be finite and nonnegative')
    if args.tensor_parallel_size < 1:
        cli.error('--tensor-parallel-size must be positive')
    if args.context_tokens <= 32768:
        cli.error('--context-tokens must exceed the 32768-token response budget')
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError('Duplicate conditions')
    run = run_directory(args.run)
    manifest, skills, memory_path, memories = prepare(run, args.conditions, args.question_ids, args.skills_dir, args.memories, args.memory_variant)
    plan = {'run': str(run), 'conditions': args.conditions, 'questions': [t['question_id'] for t in manifest['tasks']],
            'policy': 'full_history', 'max_turns': 6, 'response_tokens': 32768, 'context_tokens': args.context_tokens,
            'temperature': args.temperature, 'seed': 0, 'dtype': 'bfloat16', 'max_num_seqs': 1,
            'tool_reminder': QWEN_TOOL_REMINDER,
            'tensor_parallel_size': args.tensor_parallel_size,
            'prompt_source': 'exact saved original initial_messages; only requested privileged prefix is added',
            'tool': 'native frame_select; MM:SS; original EVA extractor/fallback',
            'skill_sections': 'all general + exact category + all common mistakes',
            'memory_policy': 'successful source: full contextual_description and content; failed source: own mistakes only',
            'scope': 'in-sample same-question diagnostic; not held-out'}
    plan['memory_variant'] = args.memory_variant
    if args.memory_variant == VARIANT:
        plan.update(memory_policy='all source trajectories: inspection plans and essential cautions; legacy targets supported',
                    prompt_source='exact saved original initial_messages; only requested inspection guidance is added')
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID') or not args.output or not args.model_dir:
        raise RuntimeError('Use submit.py eval --submit with the local model environment')
    output = args.output.resolve()
    if any(output.is_relative_to(Path(p)) for p in ('/project', '/home')):
        raise ValueError('Use scratch job I/O; worker publishes back to the same data run')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', plan)
    write(output / 'evaluation_manifest.json', manifest)
    write(output / 'selected_memories.json', list(memories.values()))
    failures = []
    for condition in args.conditions:
        cmd = [sys.executable, '-B', '-m', 'eval.qwen.backend',
               '--manifest', str(output / 'evaluation_manifest.json'), '--condition', CONDITIONS[condition],
               '--model-dir', str(args.model_dir), '--output', str(output / condition),
               '--context-tokens', str(args.context_tokens), '--response-tokens', '32768', '--max-turns', '6',
               '--temperature', str(args.temperature), '--tensor-parallel-size', str(args.tensor_parallel_size)]
        if condition == 'base_skills_mistakes':
            cmd += ['--skills-dir', str(skills)]
        if condition == 'base_memory':
            cmd += ['--memories', str(output / 'selected_memories.json')]
        result = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT / 'eval')])})
        if result.returncode:
            failures.append(condition)
    write(output / 'status.json', {'state': 'failed' if failures else 'completed', 'failed_conditions': failures})
    if failures:
        raise RuntimeError('Evaluation failures: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
