"""Saved conversations: one JSON file per chat in .chats/. (Phase 12)

A chat is the list of turns the app showed: the user's question, the question
that was searched (a follow-up rewritten to stand alone), and the full answer
with its citations, checks and the passages it was given. The passages are
saved too, so an old chat still shows its sources after the documents are
re-indexed or removed.

Chats stay on this computer (.chats/ is git-ignored) but contain quotes from
the documents, so treat the folder like the documents themselves.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from citerag import config
from citerag.generator import Answer, Citation, Segment
from citerag.store import Hit

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
TITLE_CHARS = 60
_ID_RE = re.compile(r"[0-9a-f]{12}")  # ids come from the URL: only these ever touch the disk


@dataclass
class Turn:
    question: str  # as the user typed it
    search: str  # what was searched and answered: the question, or a follow-up rewritten to stand alone
    answer: Answer


@dataclass
class Chat:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    created: str = field(default_factory=lambda: _now())
    updated: str = ""

    @property
    def title(self) -> str:
        if not self.turns:
            return "New chat"
        q = " ".join(self.turns[0].question.split())
        return q if len(q) <= TITLE_CHARS else q[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"

    def history(self) -> list[tuple[str, str]]:
        """(searched question, answer text) pairs, for rewriting the next follow-up."""
        return [(t.search, t.answer.text) for t in self.turns]


class ChatStore:
    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder or config.CHATS_DIR)

    def save(self, chat: Chat) -> None:
        chat.updated = _now()
        data = {"version": FORMAT_VERSION, "id": chat.id, "created": chat.created, "updated": chat.updated,
                "turns": [{"question": t.question, "search": t.search, "answer": answer_to_dict(t.answer)}
                          for t in chat.turns]}
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path(chat.id)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(path)

    def load(self, chat_id: str) -> Chat | None:
        """The chat, or None if the id is malformed, unknown, or the file can't be read."""
        if not _ID_RE.fullmatch(chat_id or ""):
            return None
        path = self._path(chat_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
            turns = [Turn(t["question"], t["search"], answer_from_dict(t["answer"])) for t in data["turns"]]
            return Chat(data["id"], turns, data["created"], data.get("updated", ""))
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("chat %s can't be read (%s)", chat_id, e)
            return None

    def list(self, limit: int = 30) -> list[tuple[str, str]]:
        """(id, title) of saved chats, most recently updated first."""
        chats = [c for c in (self.load(p.stem) for p in self.folder.glob("*.json")) if c is not None]
        chats.sort(key=lambda c: c.updated or c.created, reverse=True)
        return [(c.id, c.title) for c in chats[:limit]]

    def delete(self, chat_id: str) -> None:
        if _ID_RE.fullmatch(chat_id or ""):
            self._path(chat_id).unlink(missing_ok=True)

    def _path(self, chat_id: str) -> Path:
        return self.folder / f"{chat_id}.json"


def answer_to_dict(answer: Answer) -> dict:
    return dataclasses.asdict(answer)


def answer_from_dict(d: dict) -> Answer:
    from citerag.validator import Check, Validation

    v = d.get("validation")
    validation = None if v is None else Validation(
        [Check(**c) for c in v["checks"]], list(v.get("invalid_markers", [])),
        [Check(**c) for c in v.get("removed", [])])
    return Answer(
        question=d["question"], text=d["text"],
        segments=[Segment(**s) for s in d["segments"]],
        citations=[Citation(**c) for c in d["citations"]],
        hits=[Hit(**h) for h in d["hits"]],
        found=d["found"], stop_reason=d.get("stop_reason"), model=d.get("model"),
        usage=d.get("usage") or {}, invalid_markers=list(d.get("invalid_markers", [])),
        validation=validation,
    )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
