"""Compatibility entry point for the matched Qwen evaluations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.qwen.run import main, prepare

if __name__ == "__main__":
    main()
