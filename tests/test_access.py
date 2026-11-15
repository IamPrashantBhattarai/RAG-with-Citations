from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from citerag.access import (LOCAL_USER, AccessError, Limits, Policy, Registrations, ScopedRetriever, User, group_of,
                            toml_entry)
from citerag.usage import NO_LIMIT, Quota

POLICY = """
[[users]]
email = "Boss@Example.com"
admin = true

[[users]]
email = "asha@example.com"
groups = ["hr"]

[[users]]
email = "nobody@example.com"
"""


def write(tmp_path, text):
    path = tmp_path / "access.toml"
    path.write_text(text)
    return path


def test_listed_emails_get_in_with_their_groups(tmp_path):
    policy = Policy.load(write(tmp_path, POLICY))
    assert policy.user(" BOSS@example.com ").admin  # emails are matched without case or spaces
    asha = policy.user("asha@example.com")
    assert asha == User("asha@example.com", False, frozenset({"hr"}))
    assert policy.user("stranger@example.com") is None and policy.user(None) is None


def test_who_reads_what():
    asha, nobody, boss = User("a@x.io", groups=frozenset({"hr"})), User("n@x.io"), User("b@x.io", admin=True)
    docs = ["hr/policy.pdf", "finance/budget.xlsx", "public/guide.pdf", "loose.pdf", "hr/old/2019.pdf"]
    assert asha.readable(docs) == ["hr/policy.pdf", "public/guide.pdf", "hr/old/2019.pdf"]
    assert nobody.readable(docs) == ["public/guide.pdf"]  # everyone signed in reads public
    assert boss.readable(docs) == docs and LOCAL_USER.readable(docs) == docs
    assert group_of("loose.pdf") is None and group_of("hr/old/2019.pdf") == "hr"  # loose files: admins only


@pytest.mark.parametrize("text, message", [
    ('[[users]]\nemail = "not-an-email"', "isn't an email address"),
    ('[[users]]\nemail = "a@x.io"\ngroups = "hr"', "groups must be a list"),
    ('[[users]]\nemail = "a@x.io"\ngroups = ["../etc"]', "groups must be a list"),
    ('[[users]]\nemail = "a@x.io"\nadmin = "yes"', "admin must be true or false"),
    ('[[users]]\nemail = "a@x.io"\nrole = "admin"', "only admin, daily_questions, email, groups"),
    ('[[users]]\nemail = "a@x.io"\n[[users]]\nemail = "A@x.io"', "listed twice"),
    ('allow_everyone = true', "unknown setting"),
    ('[[users]\nemail = "a@x.io"', "can't be read"),
])
def test_mistakes_in_the_access_list_are_explained(tmp_path, text, message):
    with pytest.raises(AccessError, match=message):
        Policy.load(write(tmp_path, text))


def test_a_missing_access_list_says_what_to_do(tmp_path):
    with pytest.raises(AccessError, match="copy access.example.toml"):
        Policy.load(tmp_path / "access.toml")


class RecordingRetriever:
    """Returns `hits` whatever it is asked: a retriever that ignored the filter."""

    def __init__(self, hits):
        self.hits, self.calls = hits, []
        self.reranker = "RERANKER"

    def retrieve(self, query, k=5, sources=None, held_back=None):
        self.calls.append(sources)
        return self.hits


def hit(source):
    return NS(source=source)


def test_scoped_search_only_ever_sees_allowed_documents():
    inner = RecordingRetriever([hit("hr/policy.pdf"), hit("finance/budget.xlsx"), hit("public/guide.pdf")])
    scoped = ScopedRetriever(inner, ["hr/policy.pdf", "public/guide.pdf"])
    assert [h.source for h in scoped.retrieve("q")] == ["hr/policy.pdf", "public/guide.pdf"]  # leak dropped
    assert inner.calls == [["hr/policy.pdf", "public/guide.pdf"]]  # and never asked for more
    # "Search in" narrows within what's allowed, never beyond it.
    scoped.retrieve("q", sources=["public/guide.pdf", "finance/budget.xlsx"])
    assert inner.calls[-1] == ["public/guide.pdf"]
    assert scoped.reranker == "RERANKER"  # other attributes pass through


def test_nothing_allowed_means_nothing_searched():
    inner = RecordingRetriever([hit("hr/policy.pdf")])
    # The underlying retriever reads an empty or missing list as "all documents": never pass one.
    assert ScopedRetriever(inner, []).retrieve("q") == []
    assert ScopedRetriever(inner, ["hr/policy.pdf"]).retrieve("q", sources=["finance/x.pdf"]) == []
    assert inner.calls == []


def test_each_user_has_their_own_chat_key():
    assert User("a@x.io").key != User("b@x.io").key and len(User("a@x.io").key) == 16
    assert "@" not in User("a@x.io").key


def test_held_back_notes_only_name_documents_the_user_may_read():
    class Inner(RecordingRetriever):
        def retrieve(self, query, k=5, sources=None, held_back=None):
            held_back.extend([hit("hr/policy.pdf"), hit("finance/budget.xlsx")])
            return []

    held = []
    ScopedRetriever(Inner([]), ["hr/policy.pdf"]).retrieve("q", held_back=held)
    assert [h.source for h in held] == ["hr/policy.pdf"]


# -- open registration --------------------------------------------------------------


def test_open_registration_lets_unlisted_users_in_as_public_users(tmp_path):
    policy = Policy.load(write(tmp_path, "open_registration = true\n" + POLICY))
    newcomer = policy.resolve(" NewComer@Gmail.com ")
    assert newcomer == User("newcomer@gmail.com", listed=False)
    assert newcomer.readable(["hr/policy.pdf", "public/guide.pdf", "loose.pdf"]) == ["public/guide.pdf"]
    assert policy.resolve("asha@example.com").groups == {"hr"}  # listed users keep their groups
    assert policy.resolve("not an email") is None and policy.resolve(None) is None


def test_without_open_registration_only_listed_users_get_in(tmp_path):
    policy = Policy.load(write(tmp_path, POLICY))
    assert not policy.open_registration and policy.resolve("newcomer@gmail.com") is None
    with pytest.raises(AccessError, match="open_registration must be true or false"):
        Policy.load(write(tmp_path, 'open_registration = "yes"\n' + POLICY))


def test_sign_ins_are_recorded_and_unlisted_ones_are_waiting(tmp_path):
    reg = Registrations(tmp_path / "registrations.json")
    reg.seen("newcomer@gmail.com")
    first = reg.all()["newcomer@gmail.com"]["first"]
    reg.seen("newcomer@gmail.com")
    reg.seen("asha@example.com")
    assert reg.all()["newcomer@gmail.com"]["first"] == first  # the first sign-in is kept
    policy = Policy.load(write(tmp_path, POLICY))
    assert [email for email, _ in reg.unlisted(policy)] == ["newcomer@gmail.com"]


def test_a_broken_registrations_file_never_stops_a_sign_in(tmp_path):
    path = tmp_path / "registrations.json"
    path.write_text("{not json")
    Registrations(path).seen("newcomer@gmail.com")
    assert list(Registrations(path).all()) == ["newcomer@gmail.com"]


def test_the_block_to_paste_is_valid_access_toml(tmp_path):
    policy = Policy.load(write(tmp_path, toml_entry("newcomer@gmail.com")))
    assert policy.user("newcomer@gmail.com") == User("newcomer@gmail.com")


# -- blocking and limits (Phase 16) -------------------------------------------------

LIMITED = """
open_registration = true
blocked = ["Spammer@Gmail.com", "asha@example.com"]

[limits]
public = 3
member = 50
public_total = 10
per_minute = 2

[[users]]
email = "boss@example.com"
admin = true

[[users]]
email = "asha@example.com"
groups = ["hr"]

[[users]]
email = "ravi@example.com"
groups = ["finance"]
daily_questions = 500
"""


def test_blocked_people_are_kept_out_even_when_listed(tmp_path):
    policy = Policy.load(write(tmp_path, LIMITED))
    assert policy.is_blocked(" spammer@gmail.COM ") and policy.resolve("spammer@gmail.com") is None
    assert policy.user("asha@example.com") is not None  # still listed (to unblock, delete the line)...
    assert policy.resolve("asha@example.com") is None  # ...but kept out
    assert policy.resolve("newcomer@gmail.com") == User("newcomer@gmail.com", listed=False)  # others still sign up


def test_blocked_people_are_not_waiting_to_be_listed(tmp_path):
    reg = Registrations(tmp_path / "registrations.json")
    reg.seen("spammer@gmail.com")
    reg.seen("newcomer@gmail.com")
    assert [e for e, _ in reg.unlisted(Policy.load(write(tmp_path, LIMITED)))] == ["newcomer@gmail.com"]


def test_each_kind_of_user_gets_their_limits(tmp_path):
    policy = Policy.load(write(tmp_path, LIMITED))
    assert policy.quota(policy.user("boss@example.com")) == NO_LIMIT
    assert policy.quota(policy.user("ravi@example.com")) == Quota(daily=500, per_minute=2)  # their own limit
    assert policy.quota(policy.resolve("newcomer@gmail.com")) == Quota(daily=3, per_minute=2, shared=10)
    member = Policy.load(write(tmp_path, POLICY)).user("asha@example.com")
    assert Policy([member]).quota(member) == Quota(daily=100, per_minute=5)  # the defaults
    assert Limits() == Limits(public=20, member=100, public_total=200, per_minute=5)
    assert LOCAL_USER.admin and Policy([]).quota(LOCAL_USER) == NO_LIMIT


@pytest.mark.parametrize("text, message", [
    ('blocked = "spammer@gmail.com"', "blocked must be a list of email addresses"),
    ('blocked = ["spammer"]', "blocked must be a list of email addresses"),
    ('[[users]]\nemail = "a@x.io"\nblocked = ["b@x.io"]', "blocked belongs at the top of the file"),
    ('[limits]\npublic = 5\nopen_registration = true', "open_registration belongs at the top"),
    ('[limits]\npublic = -1', "public must be a whole number, 0 or more"),
    ('[limits]\nmember = 2.5', "member must be a whole number"),
    ('[limits]\nper_minute = true', "per_minute must be a whole number"),
    ('[limits]\nper_minute = 0', "per_minute must be at least 1"),
    ('[limits]\ndaily = 5', "unknown limit 'daily'"),
    ('limits = 5', "must be a section of numbers"),
    ('[[users]]\nemail = "a@x.io"\ndaily_questions = "lots"', "daily_questions must be a whole number"),
    ('[[users]]\nemail = "a@x.io"\nadmin = true\ndaily_questions = 5', "admins have no question limit"),
])
def test_mistakes_in_blocking_and_limits_are_explained(tmp_path, text, message):
    with pytest.raises(AccessError, match=message):
        Policy.load(write(tmp_path, text))


def test_the_example_access_list_loads(tmp_path):
    example = Path(__file__).parent.parent / "access.example.toml"
    policy = Policy.load(example)
    assert policy.limits == Limits() and not policy.open_registration and not policy.blocked
