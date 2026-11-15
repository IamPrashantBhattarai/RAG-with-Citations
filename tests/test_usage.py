import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from citerag import usage as usage_module
from citerag.usage import NO_LIMIT, PAUSED, SHARED_MESSAGE, Quota, Usage

NEPAL = timezone(timedelta(hours=5, minutes=45))


class Clock:
    def __init__(self, now=datetime(2026, 11, 15, 10, 0, tzinfo=NEPAL)):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def usage(tmp_path, clock):
    return Usage(tmp_path / "usage.db", clock)


def ask(usage, n, email="a@x.io", quota=Quota(daily=3)):
    return [usage.take(email, quota) for _ in range(n)]


def test_a_daily_limit_counts_down_and_resets_at_midnight(usage, clock):
    taken = ask(usage, 4)
    assert [t.allowed for t in taken] == [True, True, True, False]
    assert [t.left for t in taken] == [2, 1, 0, 0]
    assert taken[3].message == "You've used today's 3 questions. They reset at midnight."
    assert usage.status("a@x.io", Quota(daily=3)).left == 0 and not usage.status("a@x.io", Quota(daily=3)).allowed
    clock.advance(hours=13, minutes=59)  # 23:59 the same day: still out
    assert not usage.take("a@x.io", Quota(daily=3)).allowed
    clock.advance(minutes=1)  # midnight, here
    assert usage.take("a@x.io", Quota(daily=3)).left == 2


def test_each_person_has_their_own_count(usage):
    ask(usage, 3, "a@x.io")
    assert usage.take("b@x.io", Quota(daily=3)).allowed
    assert usage.today() == [("a@x.io", 3), ("b@x.io", 1)]  # busiest first


def test_status_counts_nothing(usage):
    for _ in range(5):
        status = usage.status("a@x.io", Quota(daily=3))
    assert status.allowed and status.left == 3 and status.daily == 3 and usage.today() == []


def test_a_burst_waits_until_the_oldest_question_is_a_minute_old(usage, clock):
    quota = Quota(per_minute=2)
    assert usage.take("a@x.io", quota).allowed
    clock.advance(seconds=20)
    assert usage.take("a@x.io", quota).allowed
    clock.advance(seconds=10)
    refused = usage.take("a@x.io", quota)
    assert not refused.allowed and refused.message == "That's 2 questions in a minute. Try again in 30 seconds."
    clock.advance(seconds=30)  # the first question is a minute old
    assert usage.take("a@x.io", quota).allowed


def test_sign_ups_share_one_daily_pool_and_members_are_not_in_it(usage):
    public = Quota(daily=20, shared=3)
    assert [usage.take(e, public).allowed for e in ("a@x.io", "b@x.io", "c@x.io")] == [True] * 3
    refused = usage.take("d@x.io", public)  # a new account: its own count is 0, but the pool is used up
    assert not refused.allowed and refused.message == SHARED_MESSAGE and refused.left == 20
    assert not usage.status("d@x.io", public).allowed
    assert usage.take("member@x.io", Quota(daily=100)).allowed
    assert usage.shared_today() == 3


def test_a_failed_question_is_given_back(usage):
    first = usage.take("a@x.io", Quota(daily=1))
    usage.give_back(first.ticket)
    assert usage.take("a@x.io", Quota(daily=1)).allowed
    usage.give_back(None)  # nothing to give back: fine


def test_no_limit_is_counted_but_never_refused(usage):
    assert all(t.allowed and t.left is None for t in ask(usage, 50, "boss@x.io", NO_LIMIT))
    assert usage.today() == [("boss@x.io", 50)] and usage.status("boss@x.io", NO_LIMIT).allowed


def test_zero_means_no_questions(usage):
    refused = usage.take("a@x.io", Quota(daily=0))
    assert not refused.allowed and "can't ask questions" in refused.message


def test_old_questions_are_deleted(usage, clock, tmp_path):
    ask(usage, 2)
    clock.advance(days=3)
    usage.take("b@x.io", Quota(daily=3))
    with sqlite3.connect(tmp_path / "usage.db") as db:
        assert db.execute("SELECT email FROM questions").fetchall() == [("b@x.io",)]


def test_questions_sent_at_the_same_moment_never_go_over_the_limit(tmp_path, monkeypatch):
    # Each thread has its own connection, as two browser tabs (or two app processes) would. A pause
    # between counting and recording makes a race certain: without the write lock taken before
    # counting, 18-19 of these were refused as "paused" (SQLite saw the clash); 11-16 without the pause.
    count = usage_module._count

    def slow_count(*args):
        n = count(*args)
        time.sleep(0.01)
        return n

    monkeypatch.setattr(usage_module, "_count", slow_count)
    usage, results, start = Usage(tmp_path / "usage.db"), [], threading.Barrier(20)

    def one():
        start.wait()
        results.append(usage.take("a@x.io", Quota(daily=10)).allowed)

    threads = [threading.Thread(target=one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 10 + [True] * 10 and usage.today() == [("a@x.io", 10)]


def test_when_the_counts_cant_be_kept_limited_people_wait_and_admins_carry_on(tmp_path):
    broken = tmp_path / "usage.db"
    broken.write_bytes(b"this is not a database" * 100)
    usage = Usage(broken)
    refused = usage.take("a@x.io", Quota(daily=3))
    assert not refused.allowed and refused.message == PAUSED
    assert not usage.status("a@x.io", Quota(daily=3)).allowed
    assert usage.take("boss@x.io", NO_LIMIT).allowed
