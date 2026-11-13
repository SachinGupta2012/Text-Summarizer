"""Conservative text normalization and customer-support PII masking."""

import re
import unicodedata
from dataclasses import dataclass

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{6,}\d)(?!\w)")


@dataclass(frozen=True, slots=True)
class PiiMaskingResult:
    """Masked text and the PII types that were detected."""

    text: str
    masked_types: tuple[str, ...]


def normalize_text(value: str) -> str:
    """Normalize encoding and whitespace without destroying paragraphs."""

    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def mask_customer_support_pii(value: str) -> PiiMaskingResult:
    """Mask residual email addresses and phone numbers in support dialogue."""

    masked_types: list[str] = []
    masked, email_count = _EMAIL.subn("[EMAIL]", value)
    if email_count:
        masked_types.append("email")

    masked, phone_count = _PHONE.subn("[PHONE]", masked)
    if phone_count:
        masked_types.append("phone")

    return PiiMaskingResult(text=masked, masked_types=tuple(masked_types))
