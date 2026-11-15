"""How many questions each person asks, and whether they may ask another. (Phase 16)

Every question makes one or two calls to the AI (billed to the Claude key, or
taken from the free Gemini quota the whole app shares) and runs search,
reranking and the citation checks on this computer. With open registration
anyone with a Google account can sign in, so questions are counted per person,
by the email Google verified, and limited per day and per minute.

The counts live in a small SQLite file, not in the page: the page's memory
starts again on every refresh or new tab. SQLite comes with Python, and each
question is counted in a transaction that takes the write lock before it
counts, so two questions sent at the same moment (two tabs, or two app
processes) can't both slip under a limit.

A day runs from midnight to midnight on this computer's clock.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from citerag import config

log = logging.getLogger(__name__)

KEEP_SECONDS = 2 * 24 * 3600  # older questions are deleted: only today and the last minute are ever counted
PAUSED = "Questions can't be counted right now, so they're paused. Try again in a little while."
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY,
        email TEXT NOT NULL,
        at REAL NOT NULL,         -- when, in seconds since 1970 (UTC)
        shared INTEGER NOT NULL   -- 1: counts towards the sign-ups' shared daily limit
    )""",
    "CREATE INDEX IF NOT EXISTS questions_by_email ON questions (email, at)",
    "CREATE INDEX IF NOT EXISTS questions_by_time ON questions (at)",
)


@dataclass(frozen=True)
class Quota:
    """What one person may ask. None: no limit of that kind."""

    daily: int | None = None       # their questions per day
    per_minute: int | None = None  # their questions in any 60 seconds
    shared: int | None = None      # questions per day for all sign-ups (open registration) together

    @property
    def unlimited(self) -> bool:
        return self.daily is None and self.per_minute is None and self.shared is None


NO_LIMIT = Quota()


@dataclass(frozen=True)
class Allowance:
    """Whether a person may ask (now, or another question), and what to tell them."""

    allowed: bool
    message: str = ""          # why not, in words for the person
    left: int | None = None    # their questions left today; None: no daily limit
    daily: int | None = None   # their daily limit; None: none
    ticket: int | None = None  # the counted question, to give back if answering fails


class UsageError(Exception):
    """The counts can't be read or written."""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' * (n != 1)}"


def daily_message(daily: int) -> str:
    if daily == 0:
        return "Your account can't ask questions at the moment. Ask an admin if you need to."
    return f"You've used today's {_plural(daily, 'question')}. They reset at midnight."


SHARED_MESSAGE = ("New sign-ups have used all of today's questions between them. They reset at midnight; "
                  "an admin can add you to the access list for your own allowance.")


class Usage:
    _ready: set[Path] = set()  # database files this process has set up
    _setup_lock = threading.Lock()  # Streamlit sessions are threads of one process

    def __init__(self, path: Path | None = None, clock: Callable[[], datetime] | None = None):
        self.path = Path(path or config.USAGE_FILE)
        self.clock = clock or (lambda: datetime.now().astimezone())

    # -- asking ------------------------------------------------------------------------

    def take(self, email: str, quota: Quota) -> Allowance:
        """Count one question by `email` if `quota` allows it.

        If the counts can't be read or written, a limited person is refused
        (the limits exist to cap spending) and an unlimited one is let through.
        """
        try:
            return self._take(email, quota)
        except UsageError as e:
            log.warning("question counts unavailable (%s)", e)
            return Allowance(True) if quota.unlimited else Allowance(False, PAUSED, daily=quota.daily)

    def _take(self, email: str, quota: Quota) -> Allowance:
        now = self.clock()
        t, midnight = now.timestamp(), _midnight(now)
        with self._transaction(write=True) as db:
            db.execute("DELETE FROM questions WHERE at < ?", (t - KEEP_SECONDS,))
            mine = _count(db, "email = ? AND at >= ?", (email, midnight))
            if quota.daily is not None and mine >= quota.daily:
                return Allowance(False, daily_message(quota.daily), left=0, daily=quota.daily)
            if quota.shared is not None and _count(db, "shared = 1 AND at >= ?", (midnight,)) >= quota.shared:
                return Allowance(False, SHARED_MESSAGE, left=_left(quota, mine), daily=quota.daily)
            if quota.per_minute is not None:
                recent = [row[0] for row in db.execute(
                    "SELECT at FROM questions WHERE email = ? AND at > ? ORDER BY at", (email, t - 60))]
                if len(recent) >= quota.per_minute:
                    # the question that has to drop out of the last 60 seconds before there's room again
                    wait = max(1, math.ceil(recent[len(recent) - quota.per_minute] + 60 - t))
                    return Allowance(False, f"That's {_plural(quota.per_minute, 'question')} in a minute. "
                                            f"Try again in {_plural(wait, 'second')}.",
                                     left=_left(quota, mine), daily=quota.daily)
            row = db.execute("INSERT INTO questions (email, at, shared) VALUES (?, ?, ?)",
                             (email, t, int(quota.shared is not None)))
            return Allowance(True, left=_left(quota, mine + 1), daily=quota.daily, ticket=row.lastrowid)

    def give_back(self, ticket: int | None) -> None:
        """Uncount a question that failed on our side (the AI service was down, say)."""
        if ticket is None:
            return
        try:
            with self._transaction(write=True) as db:
                db.execute("DELETE FROM questions WHERE id = ?", (ticket,))
        except UsageError as e:
            log.warning("couldn't give a question back (%s)", e)

    def status(self, email: str, quota: Quota) -> Allowance:
        """Before asking: questions left today, and why they can't ask now if they can't. Counts nothing."""
        if quota.daily is None and quota.shared is None:
            return Allowance(True)
        try:
            midnight = _midnight(self.clock())
            with self._transaction() as db:
                mine = _count(db, "email = ? AND at >= ?", (email, midnight))
                shared = _count(db, "shared = 1 AND at >= ?", (midnight,)) if quota.shared is not None else 0
        except UsageError as e:
            log.warning("question counts unavailable (%s)", e)
            return Allowance(False, PAUSED, daily=quota.daily)
        left = _left(quota, mine)
        if left == 0:
            return Allowance(False, daily_message(quota.daily), left=0, daily=quota.daily)
        if quota.shared is not None and shared >= quota.shared:
            return Allowance(False, SHARED_MESSAGE, left=left, daily=quota.daily)
        return Allowance(True, left=left, daily=quota.daily)

    # -- for admins --------------------------------------------------------------------

    def today(self) -> list[tuple[str, int]]:
        """Each person's questions since midnight, busiest first."""
        with self._transaction() as db:
            rows = db.execute("SELECT email, COUNT(*) AS n FROM questions WHERE at >= ? "
                              "GROUP BY email ORDER BY n DESC, email", (_midnight(self.clock()),))
            return [(email, n) for email, n in rows]

    def shared_today(self) -> int:
        """Questions since midnight by sign-ups not in the access list, all together."""
        with self._transaction() as db:
            return _count(db, "shared = 1 AND at >= ?", (_midnight(self.clock()),))

    # -- the file ----------------------------------------------------------------------

    @contextmanager
    def _transaction(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=10, isolation_level=None)) as db:
                self._set_up(db)
                db.execute("BEGIN IMMEDIATE" if write else "BEGIN")  # IMMEDIATE: the write lock, before counting
                try:
                    yield db
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
                db.execute("COMMIT")
        except (sqlite3.Error, OSError) as e:
            raise UsageError(f"{self.path.name}: {e}") from e

    def _set_up(self, db: sqlite3.Connection) -> None:
        """Create the table once per process. Many sessions creating a new file at the same moment can
        find it locked mid-setup, which would pause a question for no reason."""
        if self.path in Usage._ready:
            return
        with Usage._setup_lock:
            if self.path in Usage._ready:
                return
            db.execute("PRAGMA journal_mode = WAL")  # reading the counts doesn't wait for a question being counted
            for statement in _SCHEMA:
                db.execute(statement)
            Usage._ready.add(self.path)


def _midnight(now: datetime) -> float:
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _count(db: sqlite3.Connection, where: str, args: tuple) -> int:
    return db.execute(f"SELECT COUNT(*) FROM questions WHERE {where}", args).fetchone()[0]


def _left(quota: Quota, used: int) -> int | None:
    return None if quota.daily is None else max(0, quota.daily - used)
