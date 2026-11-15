"""Who may read which documents. (Phase 13)

Folders under data/ are groups: "hr/policy.pdf" belongs to the group "hr".
access.toml lists who may sign in, their groups, and who is an admin:

- everyone signed in reads the "public" group (data/public/)
- a member reads the folders of their groups
- an admin reads everything and manages documents (upload, web pages,
  remove, rescan)
- a file directly in data/, in no folder, is read by admins only: nobody
  decided who should see it, so nobody but admins does

With `open_registration = true` in access.toml, anyone with a Google account
can sign in: an email that isn't listed gets in as a public user (data/public/
only) until an admin lists it with groups. Every sign-in is recorded in a small
local file (Registrations), so admins can see who is waiting to be promoted.

Phase 16: `blocked = [...]` shuts people out, whatever else the file says, and
[limits] caps how many questions people ask (counted in citerag/usage.py):
per day for sign-ups and for members, per day for all sign-ups together, and
per minute for everyone. Admins have no limit; `daily_questions` on a user
gives that person their own daily limit.

Permissions are enforced where a document's text could leave the index: every
search goes through ScopedRetriever, which filters to the user's documents
before the LLM or the page sees anything, and checks each result again after.
An empty set of documents means "nothing", never "no filter".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from citerag import config
from citerag.usage import NO_LIMIT, Quota

log = logging.getLogger(__name__)

PUBLIC = "public"
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_GROUP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]*")
_USER_KEYS = {"email", "admin", "groups", "daily_questions"}
_SETTINGS = {"users", "open_registration", "blocked", "limits"}
_TOP_LEVEL = {"open_registration", "blocked"}  # TOML: these must come before the first [table]


class AccessError(Exception):
    """access.toml is missing or wrong; the message says what to fix."""


@dataclass(frozen=True)
class User:
    email: str
    admin: bool = False
    groups: frozenset[str] = frozenset()
    listed: bool = True  # False: signed up through open registration, not in access.toml yet
    daily_questions: int | None = None  # their own daily question limit, instead of [limits] member (Phase 16)

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


@dataclass(frozen=True)
class Limits:
    """[limits] in access.toml: how many questions people may ask. (Phase 16)"""

    public: int = 20         # per day, for each sign-up not in access.toml
    member: int = 100        # per day, for each listed user who isn't an admin
    public_total: int = 200  # per day, for all sign-ups together
    per_minute: int = 5      # for anyone but admins


def _normalise(email: str | None) -> str:
    return (email or "").strip().lower()


class Policy:
    def __init__(self, users: list[User], open_registration: bool = False, blocked=(),
                 limits: Limits | None = None):
        self.users = {u.email: u for u in users}
        self.open_registration = open_registration
        self.blocked = frozenset(_normalise(e) for e in blocked)
        self.limits = limits or Limits()

    def user(self, email: str | None) -> User | None:
        """The listed user with this email (case doesn't matter), or None."""
        return self.users.get(_normalise(email))

    def is_blocked(self, email: str | None) -> bool:
        return _normalise(email) in self.blocked

    def resolve(self, email: str | None) -> User | None:
        """Who this (Google-verified) email signs in as: the listed user; with open
        registration, an unlisted email as a public user; otherwise None, not allowed in.
        A blocked email is None, whatever else the file says."""
        if self.is_blocked(email):
            return None
        if listed := self.user(email):
            return listed
        email = _normalise(email)
        if self.open_registration and _EMAIL_RE.fullmatch(email):
            return User(email, listed=False)
        return None

    def quota(self, user: User) -> Quota:
        """How many questions `user` may ask."""
        if user.admin:
            return NO_LIMIT
        limits = self.limits
        daily = user.daily_questions if user.daily_questions is not None else (
            limits.member if user.listed else limits.public)
        return Quota(daily=daily, per_minute=limits.per_minute,
                     shared=None if user.listed else limits.public_total)

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
            if isinstance(entry, dict) and (misplaced := sorted(set(entry) & _TOP_LEVEL)):
                raise AccessError(_misplaced(where, misplaced[0]))
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
            daily = entry.get("daily_questions")
            if daily is not None and not _count_ok(daily):
                raise AccessError(f"{where}: daily_questions must be a whole number, 0 or more")
            if daily is not None and admin:
                raise AccessError(f"{where}: admins have no question limit, so daily_questions doesn't apply; "
                                  "remove it")
            users.append(User(email, admin, frozenset(groups), daily_questions=daily))
        if set(data) - _SETTINGS:
            raise AccessError(f"{path.name}: unknown setting {sorted(set(data) - _SETTINGS)[0]!r}")
        open_registration = data.get("open_registration", False)
        if not isinstance(open_registration, bool):
            raise AccessError(f"{path.name}: open_registration must be true or false")
        return cls(users, open_registration, _blocked(data.get("blocked", []), path.name),
                   _limits(data.get("limits", {}), path.name))


def _count_ok(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _misplaced(where: str, key: str) -> str:
    # In TOML a line after [limits] or [[users]] belongs to that section, however it's indented.
    return f"{where}: {key} belongs at the top of the file, above [limits] and the first [[users]]"


def _blocked(value, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(e, str) and _EMAIL_RE.fullmatch(_normalise(e))
                                              for e in value):
        raise AccessError(f'{name}: blocked must be a list of email addresses, like ["someone@gmail.com"]')
    return [_normalise(e) for e in value]


def _limits(value, name: str) -> Limits:
    where = f"{name}, [limits]"
    if not isinstance(value, dict):
        raise AccessError(f"{where} must be a section of numbers")
    if misplaced := sorted(set(value) & _TOP_LEVEL):
        raise AccessError(_misplaced(where, misplaced[0]))
    known = set(Limits.__dataclass_fields__)
    if unknown := sorted(set(value) - known):
        raise AccessError(f"{where}: unknown limit {unknown[0]!r}; the limits are {', '.join(sorted(known))}")
    for key, n in value.items():
        if not _count_ok(n):
            raise AccessError(f"{where}: {key} must be a whole number, 0 or more")
    if value.get("per_minute") == 0:
        raise AccessError(f"{where}: per_minute must be at least 1 (to stop someone asking, block them)")
    return Limits(**value)


class Registrations:
    """Who has signed in, and when: each email's first and last sign-in.

    A small JSON file (git-ignored: it lists people's emails), rewritten
    atomically. Admins use it to see who signed up through open registration
    and is still a public user. A file that can't be read or written never
    stops anyone signing in.
    """

    _lock = threading.Lock()  # Streamlit sessions are threads of one process

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.REGISTRATIONS_FILE)

    def all(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log.warning("%s can't be read (%s); starting a new one", self.path.name, e)
            return {}
        return data if isinstance(data, dict) else {}

    def seen(self, email: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            data = self.all()
            data[email] = {"first": (data.get(email) or {}).get("first", now), "last": now}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
                tmp.replace(self.path)
            except OSError as e:
                log.warning("couldn't record the sign-in of a user in %s (%s)", self.path.name, e)

    def unlisted(self, policy: Policy) -> list[tuple[str, dict]]:
        """Signed-up users not in access.toml yet (public only), newest first. Blocked people aren't waiting."""
        waiting = [(email, e) for email, e in self.all().items()
                   if policy.user(email) is None and not policy.is_blocked(email)]
        return sorted(waiting, key=lambda item: item[1].get("first", ""), reverse=True)


def toml_entry(email: str) -> str:
    """The access.toml block that lists a signed-up user, ready to paste."""
    return f'[[users]]\nemail = "{email}"\ngroups = []   # their folders under data/, e.g. ["hr"]'


class ScopedRetriever:
    """A retriever that only ever searches, and only ever returns, `allowed` documents.

    `sources` (the page's "Search in") can narrow the search further but never
    widen it. With nothing allowed, nothing is searched: the underlying
    retriever reads an empty or missing list as "all documents".
    """

    def __init__(self, retriever, allowed):
        self.retriever = retriever
        self.allowed = frozenset(allowed)

    def retrieve(self, query: str, k: int = config.TOP_K, sources: list[str] | None = None,
                 held_back: list | None = None):
        wanted = self.allowed if sources is None else self.allowed & set(sources)
        if not wanted:
            return []
        inner_held: list = []
        hits = self.retriever.retrieve(query, k=k, sources=sorted(wanted), held_back=inner_held)
        kept = [h for h in hits if h.source in wanted]
        if len(kept) < len(hits):  # the filter above should make this impossible
            log.error("search returned %d passage(s) outside the user's documents; dropped",
                      len(hits) - len(kept))
        if held_back is not None:  # even a held-back note must not name a document the user can't read
            held_back.extend(h for h in inner_held if h.source in wanted)
        return kept

    def __getattr__(self, name):  # store, reranker, ... as the wrapped retriever
        return getattr(self.retriever, name)
