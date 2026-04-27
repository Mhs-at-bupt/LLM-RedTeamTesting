from __future__ import annotations

import re


def redact_api_like_tokens(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9]{16,}", "sk-***REDACTED***", text)
    text = re.sub(r"[A-Za-z0-9]{24,}", "***REDACTED_TOKEN***", text)
    return text


def sanitize_public_text(text: str) -> str:
    return " ".join(text.split())[:1000]

