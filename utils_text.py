# utils_text.py
# Emoji/Unicode sanitizer for logs and CSVs
import re
import unicodedata

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
    "\U0001FA00-\U0001FA6F"  # Supplemental sets
    "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
    "\u2600-\u26FF"          # Misc symbols
    "\u2700-\u27BF"          # Dingbats
    "]+"
)

_ZWJ_VARIATION_RE = re.compile(r"[\u200D\uFE0F]")

def strip_emojis(text: str) -> str:
    if not text:
        return text
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZWJ_VARIATION_RE.sub("", normalized)
    return _EMOJI_RE.sub("", normalized)

def sanitize_for_csv(text: str) -> str:
    s = strip_emojis(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    return s
