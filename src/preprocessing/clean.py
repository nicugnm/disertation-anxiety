"""Text cleaning: encoding fixes, markdown stripping, whitespace, common Reddit quirks."""
from __future__ import annotations

import re

import ftfy

# Reddit / markdown patterns
RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RE_BARE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_REDDIT_QUOTE = re.compile(r"^&gt;.*$", flags=re.MULTILINE)
RE_HTML_ENTITIES = re.compile(r"&(amp|lt|gt|quot|#39);")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_REDDIT_BOT_NOTICE = re.compile(
    r"^\*?I am a bot.*$|^\*?This action was performed automatically.*$",
    flags=re.MULTILINE | re.IGNORECASE,
)

HTML_ENTITY_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
}


def fix_encoding(text: str) -> str:
    """Repair mojibake (e.g., â€™ → ')."""
    return ftfy.fix_text(text or "")


def strip_markdown(text: str) -> str:
    """Replace markdown links with their anchor text, drop quoted lines."""
    text = RE_MARKDOWN_LINK.sub(r"\1", text)
    text = RE_REDDIT_QUOTE.sub("", text)
    return text


def strip_urls(text: str) -> str:
    return RE_BARE_URL.sub(" [URL] ", text)


def decode_entities(text: str) -> str:
    for ent, ch in HTML_ENTITY_MAP.items():
        text = text.replace(ent, ch)
    return text


def normalize_whitespace(text: str) -> str:
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def strip_bot_notices(text: str) -> str:
    return RE_REDDIT_BOT_NOTICE.sub("", text)


def clean_text(title: str, body: str) -> str:
    """Run the full pipeline. Concatenates title + body for downstream models."""
    title = title or ""
    body = body or ""
    full = f"{title}\n\n{body}".strip()
    full = fix_encoding(full)
    full = decode_entities(full)
    full = strip_markdown(full)
    full = strip_urls(full)
    full = strip_bot_notices(full)
    full = normalize_whitespace(full)
    return full
