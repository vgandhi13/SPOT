"""Replay the saved original messages without added context."""
from copy import deepcopy


def build_messages(task):
    return deepcopy(task["initial_messages"])
