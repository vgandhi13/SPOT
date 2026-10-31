"""Image blocks for the frames a trajectory's tool calls returned (export.py layout)."""
import base64
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ENCODED_BYTES = 24_000_000


def frame_blocks(record):
    """Return Anthropic-style text+image blocks, labeled with turn, call and timestamp."""
    root = Path(record["run_dir"]).resolve()
    blocks, encoded = [], 0
    for event in record["tool_events"]:
        for frame in event["result"].get("frames", []):
            path = (root / frame["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Frame path escapes its trajectory folder")
            raw = path.read_bytes()
            if not raw.startswith(PNG_SIGNATURE):
                raise ValueError(f"Recorded frame is missing or not PNG: {frame['file']}")
            data = base64.b64encode(raw).decode("ascii")
            encoded += len(data)
            blocks.append({"type": "text", "text": (f"Recorded frame: assistant turn {event['turn']}, "
                f"tool call {event['tool_call_id']}, timestamp {frame['timestamp']}. "
                "The following image shows this sampled frame.")})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    if encoded > MAX_ENCODED_BYTES:
        raise ValueError("Frame request exceeds the 24 MB encoded image limit")
    return blocks
