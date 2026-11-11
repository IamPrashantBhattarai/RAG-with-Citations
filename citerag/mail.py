"""The attachments of an email (.eml, or Outlook .msg), for the loader to read. (Phase 11)

Docling reads an email's headers and body but only names its attachments,
and the answer is often in the attached PDF or Word file. This module only
pulls the attachments out; loaders.load_email reads each one with the same
loader as an uploaded file.

Skipped: inline images and small images (logos and icons in signatures,
under MIN_IMAGE_BYTES), anything over MAX_ATTACHMENT_BYTES, and attachments
past the first MAX_ATTACHMENTS. The text of a forwarded email is already in
the body; its files are returned too, one level deep. An email attached as a
file (an .eml or .msg) is returned and read as an email of its own.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass
from email import policy
from pathlib import Path

MAX_ATTACHMENTS = 20
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MIN_IMAGE_BYTES = 30_000
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Attachment:
    name: str  # a safe file name: no folders, no odd characters
    data: bytes
    inline: bool = False  # shown in the body (a logo), not attached as a file


def attachments(path: Path) -> list[Attachment]:
    """The attachments worth reading, in order."""
    raw = _msg_attachments(path) if path.suffix.lower() == ".msg" else _eml_attachments(path.read_bytes())
    out = []
    for a in raw[:MAX_ATTACHMENTS]:
        image = Path(a.name).suffix.lower() in _IMAGE_SUFFIXES
        if len(a.data) > MAX_ATTACHMENT_BYTES or (image and (a.inline or len(a.data) < MIN_IMAGE_BYTES)):
            continue
        out.append(a)
    return out


def safe_name(name: str | None, index: int, default_suffix: str = "") -> str:
    """"../../etc/passwd" -> "passwd"; empty or odd names -> "attachment-2.pdf"."""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[^\w.() -]+", "_", base).strip(" .")[:120]
    return base or f"attachment-{index}{default_suffix}"


def _eml_attachments(data: bytes) -> list[Attachment]:
    """The message's own attachment parts, and those of an email forwarded inside it.

    A forwarded email (a message/rfc822 part) isn't returned itself: Docling
    already reads its text into the body (it was read twice before). Its
    files aren't in the body, so they are returned, one level deep.
    """
    msg = email.message_from_bytes(data, policy=policy.default)
    return _parts(msg, depth=0) if hasattr(msg, "iter_attachments") and msg.is_multipart() else []


def _parts(msg, depth: int) -> list[Attachment]:
    out = []
    for i, part in enumerate(msg.iter_attachments(), start=1):
        if part.get_content_type() == "message/rfc822":
            inner = part.get_payload(0) if part.is_multipart() else part.get_payload()
            if depth == 0 and hasattr(inner, "iter_attachments") and inner.is_multipart():
                out.extend(_parts(inner, depth + 1))
            continue
        payload = part.get_payload(decode=True) or b""
        inline = part.get_content_disposition() == "inline" or part["Content-ID"] is not None
        out.append(Attachment(safe_name(part.get_filename(), i), payload, inline))
    return out


def _msg_attachments(path: Path) -> list[Attachment]:
    from oxmsg import Message

    msg = Message.load(path)
    return [Attachment(safe_name(a.file_name, i), a.file_bytes or b"")
            for i, a in enumerate(msg.attachments, start=1)]
