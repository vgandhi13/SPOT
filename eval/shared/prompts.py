"""EVA's original system prefix plus explicit native-call and history guidance."""
from copy import deepcopy
import re

EVA_SYSTEM = "Use Frame Select Tool to Anlysis the video and generate an answer to the question."

FINAL_TURN_WARNING = (
    "This is your final allowed response. No further model turns will follow. "
    "Use the evidence already available and give your best final answer now, "
    "even if uncertain, as <answer>X</answer> with exactly one supplied option label. "
    "Do not call tools or request more frames."
)

# Unlike SPOT/Claude's tool this has no 32-frame cap and requires nframes,
# matching EVA's evaluator rather than its extractor's CLI default.
FRAME_TOOL = {"type": "function", "function": {
    "name": "frame_select",
    "description": "Uniformly sample frames from an interval of this question's video.",
    "parameters": {"type": "object", "properties": {
        "start_time": {"type": "string", "pattern": r"^[0-9]+:[0-5][0-9]$",
                       "description": "Video-relative start time as quoted MM:SS, e.g. 1:40."},
        "end_time": {"type": "string", "pattern": r"^[0-9]+:[0-5][0-9]$",
                     "description": "Video-relative end time as quoted MM:SS, e.g. 2:50; later than start_time."},
        "nframes": {"type": "integer", "minimum": 1, "description": "Requested number of frames; no fixed upper count cap."},
        "resize": {"type": "number", "exclusiveMinimum": 0, "default": 1.0,
                   "description": "Image scaling factor, e.g. 0.5; EVA may reduce it to fit the per-call visual-token budget."}},
        "required": ["start_time", "end_time", "nframes"], "additionalProperties": False}}}

NATIVE_GUIDANCE = """
Use the API's native frame_select function tool to inspect the video. Pass times
as quoted "MM:SS" strings relative to the video, e.g. start_time="1:40" and
end_time="2:50". Use one colon and exactly two seconds digits from 00 to 59;
minutes may exceed 59. Numeric seconds and fractional times are not accepted.
The tool uniformly samples nframes frames
in the requested interval; one frame uses its midpoint. Optional resize defaults
to 1.0. The EVA per-call visual-token fallback may reduce resolution/frame count.
You may request multiple independent calls in one response. Calls execute
sequentially, and all results arrive before your next response. There is no
separate per-turn or total tool-call count cap. Dependent calls need another turn.
Use reasoning when helpful. Finish with <answer>X</answer>, where X is a supplied
option label. A final answer ends evaluation before any calls in that response.
""".strip()

FULL_GUIDANCE = """
""".strip()

def build_messages(prompt, duration, resolution, policy):
    """Preserve EVA's system insertion/duration rounding; display time as MM:SS."""
    if policy != "full_history":
        raise ValueError("Only full_history is supported")
    if not prompt:
        raise ValueError("Empty prompt")
    messages = deepcopy(prompt)
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": EVA_SYSTEM})
    guidance = FULL_GUIDANCE
    messages[0]["content"] += "\n\n" + NATIVE_GUIDANCE + "\n\n" + guidance
    user = next((m for m in messages if m["role"] == "user"), None)
    if user is None or not isinstance(user.get("content"), str):
        raise ValueError("Require a text user question")
    raw_question = user["content"].strip()
    width, height = (int(v) for v in resolution.split("x"))
    minutes, seconds = divmod(int(max(duration - 0.5, 1)), 60)
    user["content"] = (f"Video Length: {minutes}:{seconds:02d} (MM:SS). "
                       f"Original video resolution: {min(width, height)}p. {raw_question}")
    return messages, raw_question


def extract_answer(text):
    """Same two answer regexes and precedence as eval-eva.py."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    answer = match.group(1).strip() if match else ""
    if not answer:
        match = re.search(r"[Aa]nswer[\.:\s]\s*(\w)(\W.{,100}|)$", text, re.DOTALL)
        answer = match.group(1).strip() if match else ""
    return answer
