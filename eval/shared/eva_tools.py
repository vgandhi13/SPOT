"""EVA's ffprobe, visual-budget fallback and extractor subprocess interface."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys

EXTRACTOR = Path(__file__).with_name("select_frame_fallback.py")


def probe(video):
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration,width,height", "-of", "json", str(video)],
        capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError("ffprobe failed")
    streams = json.loads(proc.stdout or "{}").get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no video streams")
    stream = streams[0]
    return float(stream["duration"]), f"{int(stream['width'])}x{int(stream['height'])}"


def estimated_tokens(width, height, nframes, resize):
    return int(nframes) * max(1, round(height * resize / 28)) * max(1, round(width * resize / 28))


def parse_mmss(value, name):
    """Validate the native-tool format before converting to extractor seconds."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+:[0-5][0-9]", value) is None:
        raise ValueError(f'{name} must be a quoted MM:SS string, e.g. "1:40"')
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def prepare_arguments(arguments, width, height, max_visual_tokens=12000, maxp=720):
    """EVA's two-stage fallback, without a maximum nframes restriction."""
    if not isinstance(arguments, dict):
        raise ValueError("arguments_not_json_object")
    if "start_time" not in arguments or "end_time" not in arguments:
        raise ValueError("no_start_or_end_time")
    if "nframes" not in arguments:
        raise ValueError("no_nframes")
    parsed = dict(arguments)
    # Validate locally too: providers don't always enforce JSON Schema patterns.
    for name in ("start_time", "end_time"):
        parsed[name] = parse_mmss(parsed[name], name)
    nframes = parsed["nframes"]
    if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes <= 0:
        raise ValueError("nframes must be a positive integer")
    resize = parsed.get("resize", 1.0)
    if resize is None:
        resize = 1.0
    if isinstance(resize, bool) or not isinstance(resize, (int, float)) or not math.isfinite(resize) or resize <= 0:
        raise ValueError("resize must be positive and finite")
    current = estimated_tokens(width, height, nframes, resize)
    if current > max_visual_tokens:
        resize = min(resize, min(1.0, maxp / min(width, height)))
        current = estimated_tokens(width, height, nframes, resize)
        if current > max_visual_tokens:
            r = (max_visual_tokens / current) ** (1 / 3)
            nframes = max(1, int(nframes * r))
            resize *= r
    parsed.update(nframes=nframes, resize=resize)
    return parsed, estimated_tokens(width, height, nframes, resize)


class EVAFrameTool:
    def __init__(self, video, output, *, max_visual_tokens=12000, maxp=720, fallback=True):
        self.video = Path(video).resolve()
        self.output = Path(output).resolve()
        self.duration, self.resolution = probe(self.video)
        self.width, self.height = map(int, self.resolution.split("x"))
        self.max_visual_tokens, self.maxp, self.fallback = max_visual_tokens, maxp, fallback

    def execute(self, arguments):
        parsed, tokens = prepare_arguments(arguments, self.width, self.height,
                                           self.max_visual_tokens, self.maxp)
        filtered = ({k: v for k, v in parsed.items()
                     if k in {"start_time", "end_time", "resize", "nframes"}}
                    if self.fallback else parsed)
        command = [sys.executable, str(EXTRACTOR), "--video-path", str(self.video)]
        for name, value in filtered.items():
            command.extend([f"--{name}", str(value)])
        command.extend(["--save-root", str(self.output), "--factor", "28"])
        if self.fallback:
            command.append("--clamp")
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        match = re.search(r"from (.*?)/frame_(\d+)\.([pngje]*) to (.*?)/frame_(\d+)\.([pngje]*)", proc.stdout)
        if proc.returncode or match is None:
            raise RuntimeError("call_frame_select_failed")
        folder, start, ext, end = match.group(1), int(match.group(2)), match.group(3), int(match.group(5))
        # Deliberately preserve evaluator timestamps (integer interpolation of
        # requested bounds), rather than silently substituting decoder PTS.
        step = (parsed["end_time"] - parsed["start_time"]) / max(1, end - start)
        frames = [{"path": str(Path(folder) / f"frame_{i:04d}.{ext}"),
                   "timestamp_seconds": int(parsed["start_time"] + (i - start) * step)}
                  for i in range(start, end + 1)]
        return {"requested_arguments": arguments, "argument_time_format": "MM:SS",
                "effective_arguments": parsed, "effective_time_unit": "seconds",
                "estimated_visual_tokens": tokens, "timestamp_policy": "eva_requested_bounds_integer_interpolation",
                "frames": frames}
