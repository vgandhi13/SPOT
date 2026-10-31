"""Explicit model transport; prompts are independent of the backend."""


def call_model(messages, args, cache, record=None):
    if args.backend == 'claude':
        from .claude_code import call_model as send
    elif args.backend == 'openrouter':
        from .openrouter import call_model as send
    else:
        raise ValueError('Unknown model backend')
    return send(messages, args, cache, record=record)
