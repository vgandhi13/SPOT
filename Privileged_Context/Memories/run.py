"""Create privileged memories from exported Glimpse trajectories (run_<i> folders).

Without --execute, prints the plan and makes no model calls. Rerunning with the same
--output skips memories already made, so a run stopped by a rate limit can be resumed. With --include-frames,
each memory request also carries the frames the agent received (needs an image-capable model).

  python -m Privileged_Context.Memories.run --holmes-json <train json> --output <scratch dir> \\
      --trajectories <data/glimpse_correct/q0012/run_2> [...] [--include-frames] [--backend claude] --execute
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from ..openrouter import ENDPOINT, MODEL, check_free_route, validate_endpoint
from ..util import read, write
from . import pipeline
from .records import holmes_index, load_record


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--trajectories', type=Path, nargs='+', default=[], help='run_<i> folders from export.py')
    cli.add_argument('--trajectory-list', type=Path, help='File with one run_<i> folder per line')
    cli.add_argument('--holmes-json', type=Path, required=True, help='Video-Holmes split JSON with question rows')
    cli.add_argument('--output', type=Path, help='Output folder (not under /home or /project); existing memories in it are skipped')
    cli.add_argument('--backend', choices=('openrouter', 'claude'), default='openrouter')
    cli.add_argument('--model', help='Default: the free Nemotron 3 Ultra route, or claude-sonnet-4-6 for Claude')
    cli.add_argument('--allow-paid', action='store_true', help='Allow non-free pricing for an explicitly selected model')
    cli.add_argument('--include-frames', action='store_true', help='Attach the frames each tool call returned')
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--request-interval', type=float, default=4.0)
    cli.add_argument('--timeout', type=int, default=900)
    cli.add_argument('--max-attempts', type=int, default=10)
    cli.add_argument('--max-tokens', type=int, default=32768)
    cli.add_argument('--temperature', type=float, default=0.0)
    cli.add_argument('--claude-bin', default='claude')
    cli.add_argument('--endpoint', default=ENDPOINT)
    cli.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    args = cli.parse_args()
    args.model = args.model or (MODEL if args.backend == 'openrouter' else 'claude-sonnet-4-6')
    if args.backend == 'openrouter':
        validate_endpoint(args.endpoint)
    if min(args.timeout, args.max_attempts, args.max_tokens) < 1 or args.request_interval < 0:
        raise ValueError('Invalid retry/token settings')
    if args.trajectory_list:
        args.trajectories += [Path(line) for line in args.trajectory_list.read_text().split()]
    if not args.trajectories:
        raise ValueError('Give --trajectories and/or --trajectory-list')
    rows = holmes_index(args.holmes_json)
    records = [load_record(path, rows) for path in args.trajectories]
    plan = {'backend': args.backend, 'model': args.model, 'include_frames': args.include_frames,
            'evidence_mode': pipeline.WITH_FRAMES if args.include_frames else pipeline.TEXT_ONLY,
            'trajectories': [str(p) for p in args.trajectories],
            'successes': sum(r['outcome'] == 'Success' for r in records),
            'memory_requests': sum(len(pipeline.memory_jobs(r, args.include_frames)) for r in records),
            'frames': sum(len(e['result']['frames']) for r in records for e in r['tool_events'])
                      if args.include_frames else 0}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:
        raise RuntimeError('Run model calls inside a Slurm job, with --output')
    output = args.output.resolve()
    if any(output.is_relative_to(Path(p)) for p in ('/project', '/home')):
        raise ValueError('Write job output to scratch, not /home or /project')
    if args.backend == 'openrouter':
        if not os.environ.get(args.api_key_env):
            raise ValueError(f'Set {args.api_key_env} privately before running')
        route = check_free_route(args.model, allow_paid=args.allow_paid)
        if args.include_frames and 'image' not in route['input_modalities']:
            raise ValueError('The selected model route does not accept images; drop --include-frames or pick a vision model')
        args.transport_version = 'openrouter-chat-completions-v1'
        plan['route'] = route
    else:
        executable = shutil.which(args.claude_bin)
        if executable is None:
            raise ValueError('Install and authenticate Claude Code first')
        args.claude_bin = executable
        args.claude_version = subprocess.check_output([executable, '--version'], text=True, timeout=30).strip()
        status = subprocess.run([executable, 'auth', 'status', '--json'], capture_output=True, text=True, timeout=30)
        if status.returncode or not json.loads(status.stdout).get('loggedIn'):
            raise ValueError('Run claude auth login first')
        args.transport_version = args.claude_version
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'config.json'
    if config.exists() and {k: read(config).get(k) for k in ('backend', 'model', 'include_frames')} != \
            {'backend': args.backend, 'model': args.model, 'include_frames': args.include_frames}:
        raise ValueError('Output folder holds memories from different settings; use a fresh folder')
    write(output / f'config-{os.environ["SLURM_JOB_ID"]}.json', {**plan, 'transport_version': args.transport_version,
                                   'temperature': args.temperature, 'max_tokens': args.max_tokens})
    args.output = output
    try:
        if not config.exists():
            write(config, plan)
        result = pipeline.generate_memories(records, args)
        pipeline.validate_memories(read(result['path']))
        print(json.dumps(result), flush=True)
    except pipeline.QuotaExhausted as exc:
        raise SystemExit(f'Stopped: {exc}')


if __name__ == '__main__':
    main()
