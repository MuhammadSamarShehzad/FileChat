import re

def _fix_char_spaced(s: str) -> str:
    """Collapse sequences like 'g i t h u b' → 'github'."""
    if not s:
        return ""
    return re.sub(r'(?:[A-Za-z0-9]\s){2,}[A-Za-z0-9]', lambda m: m.group(0).replace(' ', ''), s)

def clean_for_indexing(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\xad", "")  # zero‑width + soft hyphen
    text = _fix_char_spaced(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text