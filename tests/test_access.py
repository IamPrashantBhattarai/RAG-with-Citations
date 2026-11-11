from types import SimpleNamespace as NS

import pytest

from citerag.access import LOCAL_USER, AccessError, Policy, ScopedRetriever, User, group_of

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
    ('[[users]]\nemail = "a@x.io"\nrole = "admin"', "only admin, email, groups"),
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

    def retrieve(self, query, k=5, sources=None):
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
