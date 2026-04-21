from __future__ import annotations

import re
from html.parser import HTMLParser

import requests


_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


def load_url(url: str, *, timeout: float = 20.0, user_agent: str = "distillery/0.1") -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()
    body = r.text

    if "text/plain" in content_type or url.lower().endswith((".txt", ".md")):
        return body

    parser = _TextExtractor()
    parser.feed(body)
    text = parser.text
    return re.sub(r"\n{3,}", "\n\n", text).strip()
