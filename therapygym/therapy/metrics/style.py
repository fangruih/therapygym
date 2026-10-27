from __future__ import annotations

import re
from typing import Any, Dict, List


# Rough-but-practical emoji detector without external deps.
# Covers most common emoji blocks.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols, etc.
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "]"
)


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def compute_style_metrics(text: str) -> Dict[str, float]:
    """
    Compute lightweight "tone/style" diagnostics for therapist outputs.
    Returns numeric values for easy logging.
    """
    if not text:
        return {
            "style/chars": 0.0,
            "style/lines": 0.0,
            "style/newlines": 0.0,
            "style/emoji_count": 0.0,
            "style/bullet_lines": 0.0,
            "style/all_caps_words": 0.0,
            "style/avg_line_len": 0.0,
            "style/max_line_len": 0.0,
        }

    lines: List[str] = text.splitlines()
    newline_count = _safe_int(text.count("\n"))
    emoji_count = len(_EMOJI_RE.findall(text))

    bullet_lines = 0
    for ln in lines:
        s = ln.lstrip()
        if s.startswith(("-", "*", "•", "👉")):
            bullet_lines += 1
        elif re.match(r"^\d+\.", s):
            bullet_lines += 1

    # Count ALL-CAPS words (performative emphasis proxy).
    all_caps_words = 0
    for w in re.findall(r"\b[A-Za-z]{2,}\b", text):
        if w.isupper():
            all_caps_words += 1

    line_lens = [len(ln) for ln in lines] if lines else [0]
    avg_line_len = sum(line_lens) / max(1, len(line_lens))
    max_line_len = max(line_lens) if line_lens else 0

    return {
        "style/chars": float(len(text)),
        "style/lines": float(len(lines)),
        "style/newlines": float(newline_count),
        "style/emoji_count": float(emoji_count),
        "style/bullet_lines": float(bullet_lines),
        "style/all_caps_words": float(all_caps_words),
        "style/avg_line_len": float(avg_line_len),
        "style/max_line_len": float(max_line_len),
    }


