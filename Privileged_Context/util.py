"""JSON helpers shared by the transports and every kind of privileged context."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
