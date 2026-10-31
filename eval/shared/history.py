"""Complete replay of all messages, images, and returned reasoning."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

REASONING_FIELDS = ("reasoning", "reasoning_details", "reasoning_content")
DEFAULT_TOOL_REMINDER = "If more information is needed, call the frame selection tool again."


def compact(value):
    """Record request structure and image identity without duplicating base64."""
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            url = value["image_url"]["url"]
            return {"type": "image_url", "image_url": {"url": "[saved PNG payload]"},
                    "data_url_sha256": hashlib.sha256(url.encode()).hexdigest()}
        return {k: compact(v) for k, v in value.items()}
    return value


class History:
    def __init__(self, messages, policy, *, tool_reminder=None):
        if policy != "full_history":
            raise ValueError("Only full_history is supported")
        self.messages = deepcopy(messages)
        self.tool_reminder = DEFAULT_TOOL_REMINDER if tool_reminder is None else tool_reminder

    def assistant(self, raw):
        message = {"role": "assistant", "content": raw.get("content")}
        for field in ("tool_calls", *REASONING_FIELDS):
            if field in raw:
                message[field] = deepcopy(raw[field])
        self.messages.append(message)
        return message

    def tool(self, call_id, result, frame_dir):
        # Absolute frame paths leak workspace names and video IDs to the model; show the
        # relative names that eval/glimpse/export.py gives the frames instead.
        visible = {**result, "frames": [{"path": f"{frame_dir}/frame_{k:02d}.png",
                                         "timestamp_seconds": f["timestamp_seconds"]}
                                        for k, f in enumerate(result.get("frames", []))]}
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "content": json.dumps(visible, ensure_ascii=False)})

    def observation(self, call_id, result):
        parts = [{"type": "text", "text": f"<tool_response>Result {call_id}:"}]
        for i, frame in enumerate(result.get("frames", [])):
            sec = frame["timestamp_seconds"]
            minutes, seconds = divmod(sec, 60)
            parts.append({"type": "text", "text": f"Frame {i} at [{minutes:02d}:{seconds:02d}]: "})
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(
                    Path(frame["path"]).read_bytes()).decode("ascii")}})
        parts.append({"type": "text", "text": "</tool_response>"})
        message = {"role": "user", "content": parts}
        self.messages.append(message)

    def reminder(self, question):
        text = self.tool_reminder + "\n\n"
        self.messages.append({"role": "user", "content": text + f"Question: {question}"})

    def payload_messages(self):
        return deepcopy(self.messages)
