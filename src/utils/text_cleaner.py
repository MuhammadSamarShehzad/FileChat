import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)   # normalize unicode
    text = text.replace("\u202f", " ")           # fix non-breaking spaces
    return text.strip(" .")                      # remove trailing dots/spaces
