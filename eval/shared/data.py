"""Read EVA JSONL, the existing prepared bundle, or a single local MCQ task."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_task(question, video, index):
    options = question.get("options")
    if not isinstance(options, dict) or not options or question.get("answer") not in options:
        raise ValueError("Task requires labeled options and a matching answer label")
    if not isinstance(question.get("question"), str):
        raise ValueError("Task requires question text")
    content = question["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    return {"index": index, "video": str(Path(video).resolve()),
            "prompt": [{"role": "user", "content": content}], "gt": question["answer"],
            "video_id": question.get("video_id"), "question_id": question.get("question_id"),
            "question_type": question.get("question_type")}


def holmes_tasks(args):
    """Official Video-Holmes split JSON, optionally restricted to listed question IDs."""
    if not args.video_root:
        raise ValueError("--holmes-json requires --video-root")
    rows = json.loads(args.holmes_json.read_text())
    if args.question_ids:
        wanted = {int(v) for v in args.question_ids.read_text().split()}
        rows = [r for r in rows if r["Question ID"] in wanted]
    if args.exclude_question_ids and args.exclude_question_ids.is_file():
        excluded = {int(v) for v in args.exclude_question_ids.read_text().split()}
        rows = [r for r in rows if r["Question ID"] not in excluded]
    selected = rows[args.sample_offset:args.sample_offset + args.samples]
    if not selected:
        raise ValueError("No Video-Holmes questions selected")
    return [question_task({"question": r["Question"], "options": r["Options"], "answer": r["Answer"],
                           "video_id": r["video ID"], "question_id": r["Question ID"],
                           "question_type": r["Question Type"]},
                          args.video_root / f"{r['video ID']}.mp4", i)
            for i, r in enumerate(selected, args.sample_offset)]


def load_tasks(args):
    if getattr(args, "holmes_json", None):
        return holmes_tasks(args)
    if args.task:
        if args.sample_offset or args.samples != 1:
            raise ValueError("--task supports exactly one sample at offset zero")
        question = json.loads(args.task.read_text())
        video = Path(question["video_path"])
        if not video.is_absolute():
            video = args.task.resolve().parent / video
        return [question_task(question, video, 0)]
    if args.data:
        root = args.data.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported bundle manifest")
        verified = set()

        def checked(relative):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or relative not in manifest["files"]:
                raise ValueError("File outside bundle manifest")
            if relative not in verified:
                info = manifest["files"][relative]
                if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                    raise ValueError(f"Bundle checksum mismatch: {relative}")
                verified.add(relative)
            return path

        questions = json.loads(checked("questions.json").read_text())
        selected = questions[args.sample_offset:args.sample_offset + args.samples]
        if len(selected) != args.samples:
            raise ValueError("Insufficient bundle questions")
        tasks = []
        for i, question in enumerate(selected, args.sample_offset):
            vid = question["video_id"]
            if not isinstance(vid, str) or not vid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in vid):
                raise ValueError("Invalid video ID")
            tasks.append(question_task(question, checked(f"videos_cropped/{vid}.mp4"), i))
        return tasks
    if not args.video_root:
        raise ValueError("--eva-jsonl requires --video-root")
    tasks = []
    with args.eva_jsonl.open() as stream:
        for i, line in enumerate(stream):
            if i >= args.sample_offset + args.samples:
                break
            if i < args.sample_offset:
                continue
            row = json.loads(line)
            tasks.append({"index": row.get("index", i), "suite_sample_index": i,
                          "video": str((args.video_root / row["videos"][0]).resolve()),
                          "videos": row["videos"], "prompt": row["prompt"],
                          "gt": row.get("reward_model", {}).get("ground_truth")})
    if len(tasks) != args.samples:
        raise ValueError("Insufficient EVA JSONL questions")
    return tasks
