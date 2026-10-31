"""Package an exported Glimpse dataset (see export.py) for the Hugging Face Hub.

Output folder:
  README.md                               dataset card (copied from --card)
  <split>/q<ID>/run_<i>/trajectory.json   with the local `source` field removed
  <split>/q<ID>/run_<i>/raw_trace.jsonl
  frames/<split>-<NNNNN>.tar              frame PNGs, members named q<ID>/run_<i>/frames/...
  index.json, solved_ids.txt
Frames are sharded into tars to keep the repo's file count low; extracting a split's
shards inside <split>/ restores the export layout.
--redact replaces literal strings (e.g. local path components the model quoted from
tool results) with "[redacted]" in trajectory.json and raw_trace.jsonl.
"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", type=Path, help="Dataset prefix used by export.py, e.g. data/glimpse")
    parser.add_argument("output", type=Path, help="Fresh staging folder to create")
    parser.add_argument("--card", type=Path, required=True, help="Dataset card markdown, saved as README.md")
    parser.add_argument("--shard-gb", type=float, default=2.0)
    parser.add_argument("--redact", action="append", default=[], help="Literal string to redact; repeatable")
    args = parser.parse_args()
    prefix, out = args.prefix.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    shutil.copy2(args.card, out / "README.md")
    shutil.copy2(prefix.with_name(prefix.name + "_index.json"), out / "index.json")
    shutil.copy2(prefix.with_name(prefix.name + "_solved_ids.txt"), out / "solved_ids.txt")
    limit = int(args.shard_gb * 1e9)
    redactions = sorted(args.redact, key=len, reverse=True)

    def redact(text):
        for value in redactions:
            text = text.replace(value, "[redacted]")
        return text

    for split in ("correct", "incorrect"):
        source = prefix.with_name(f"{prefix.name}_{split}")
        shard, size, count, tar = 0, 0, 0, None
        for traj in sorted(source.glob("q*/run_*/trajectory.json")):
            run = traj.parent
            rel = run.relative_to(source)
            target = out / split / rel
            target.mkdir(parents=True)
            trajectory = json.loads(traj.read_text())
            trajectory.pop("source", None)
            (target / "trajectory.json").write_text(
                redact(json.dumps(trajectory, indent=2, ensure_ascii=False)) + "\n")
            (target / "raw_trace.jsonl").write_text(redact((run / "raw_trace.jsonl").read_text()))
            for frame in sorted(run.glob("frames/turn_*/call_*/frame_*.png")):
                if tar is None or size >= limit:
                    if tar is not None:
                        tar.close()
                        shard += 1
                    tar, size = tarfile.open(out / "frames" / f"{split}-{shard:05d}.tar", "w"), 0
                tar.add(frame, arcname=str(frame.relative_to(source)))
                size += frame.stat().st_size
            count += 1
        if tar is not None:
            tar.close()
        print(json.dumps({"split": split, "runs": count, "frame_shards": shard + 1 if tar else 0}), flush=True)


if __name__ == "__main__":
    main()
