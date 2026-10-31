"""Run EVA with native API tools and complete message/reasoning history."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.shared.runner import main

if __name__ == "__main__":
    main("full_history")
