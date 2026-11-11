"""Who may read which documents. (Phase 13)

Folders under data/ are groups: "hr/policy.pdf" belongs to the group "hr".
access.toml lists who may sign in, their groups, and who is an admin:

- everyone signed in reads the "public" group (data/public/)
- a member reads the folders of their groups
- an admin reads everything and manages documents (upload, web pages,
  remove, rescan)
- a file directly in data/, in no folder, is read by admins only: nobody
  decided who should see it, so nobody but admins does

Permissions are enforced where a document's text could leave the index: every
search goes through ScopedRetriever, which filters to the user's documents
before the LLM or the page sees anything, and checks each result again after.
An empty set of documents means "nothing", never "no filter".
"""

from __future__ import annotations

import hashlib
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from citerag import config

log = logging.getLogger(__name__)

PUBLIC = "public"
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_GROUP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]*")
_USER_KEYS = {"email", "admin", "groups"}


class AccessError(Exception):
    """access.toml is missing or wrong; the message says what to fix."""


@dataclass(frozen=True)
class User:
    email: str
    admin: bool = False
    groups: frozenset[str] = frozenset()

    def can_read(self, source: str) -> bool:
        group = group_of(source)
        if self.admin:
            return True
        return group is not None and (group == PUBLIC or group in self.groups)

    def readable(self, sources) -> list[str]:
        return [s for s in sources if self.can_read(s)]

    @property
    def key(self) -> str:
        """A stable, file-name-safe id: chats are kept per user under it."""
        return hashlib.sha256(self.email.encode()).hexdigest()[:16]


# The single user of a local install without sign-in: sees and manages everything, as before Phase 13.
LOCAL_USER = User("local", admin=True)


def group_of(source: str) -> str | None:
    """"hr/policy.pdf" -> "hr"; "notes.pdf" (in no folder) -> None."""
    parts = PurePosixPath(source).parts
    return parts[0] if len(parts) > 1 else None


class Policy:
    def __init__(self, users: list[User]):
        self.users = {u.email: u for u in users}

    def user(self, email: str | None) -> User | None:
        """The listed user with this email (case doesn't matter), or None: not allowed in."""
        return self.users.get((email or "").strip().lower())

    @classmethod
    def load(cls, path: Path | None = None) -> Policy:
        path = Path(path or config.ACCESS_FILE)
        if not path.is_file():
            raise AccessError(f"No access list at {path.name}: copy access.example.toml to {path.name} "
                              "and list who may sign in.")
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise AccessError(f"{path.name} can't be read: {e}") from e
        users = []
        for i, entry in enumerate(data.get("users", []), start=1):
            where = f"{path.name}, user {i}"
            if not isinstance(entry, dict) or set(entry) - _USER_KEYS:
                raise AccessError(f"{where}: only {', '.join(sorted(_USER_KEYS))} are allowed")
            email = str(entry.get("email", "")).strip().lower()
            if not _EMAIL_RE.fullmatch(email):
                raise AccessError(f"{where}: {entry.get('email')!r} isn't an email address")
            groups = entry.get("groups", [])
            if not isinstance(groups, list) or not all(isinstance(g, str) and _GROUP_RE.fullmatch(g) for g in groups):
                raise AccessError(f"{where}: groups must be a list of folder names, like [\"hr\", \"finance\"]")
            admin = entry.get("admin", False)
            if not isinstance(admin, bool):
                raise AccessError(f"{where}: admin must be true or false")
            if email in {u.email for u in users}:
                raise AccessError(f"{where}: {email} is listed twice")
            users.append(User(email, admin, frozenset(groups)))
        if set(data) - {"users"}:
            raise AccessError(f"{path.name}: unknown setting {sorted(set(data) - {'users'})[0]!r}")
        return cls(users)


class ScopedRetriever:
    """A retriever that only ever searches, and only ever returns, `allowed` documents.

    `sources` (the page's "Search in") can narrow the search further but never
    widen it. With nothing allowed, nothing is searched: the underlying
    retriever reads an empty or missing list as "all documents".
    """

    def __init__(self, retriever, allowed):
        self.retriever = retriever
        self.allowed = frozenset(allowed)

    def retrieve(self, query: str, k: int = config.TOP_K, sources: list[str] | None = None):
        wanted = self.allowed if sources is None else self.allowed & set(sources)
        if not wanted:
            return []
        hits = self.retriever.retrieve(query, k=k, sources=sorted(wanted))
        kept = [h for h in hits if h.source in wanted]
        if len(kept) < len(hits):  # the filter above should make this impossible
            log.error("search returned %d passage(s) outside the user's documents; dropped",
                      len(hits) - len(kept))
        return kept

    def __getattr__(self, name):  # store, reranker, ... as the wrapped retriever
        return getattr(self.retriever, name)
