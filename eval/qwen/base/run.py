"""Run or preview the base Qwen evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.qwen.run import main

if __name__ == "__main__":
    main("base")
