from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import streamlit as st
from pypdf import PdfReader, PdfWriter
from streamlit.testing.v1 import AppTest

from citerag import auth, config, ui
from citerag.generator import GenerationError
from citerag.store import IndexReport
from citerag.chats import Chat, ChatStore, Turn
from citerag.validator import Check, Validation, Validator
from tests.test_validator import HITS, FakeBackend, FakeNLI, answer, validate

APP = str(Path(__file__).parent.parent / "app.py")


# -- rendering -------------------------------------------------------------------------


def test_warned_statement_is_highlighted_with_its_reasons():
    md = ui.answer_markdown(validate("The pass marks are 45 [1]. Unit VII covers PHP [2]."))
    assert md.startswith("The pass marks are 45 <sup>[1]</sup>. ")
    assert '<span style="' in md and 'title="not in the cited passage: &quot;PHP&quot;' in md
    assert md.endswith(">Unit VII covers PHP <sup>[2]</sup>.</span>")  # the whole sentence


def test_list_bullet_stays_outside_the_highlight():
    md = ui.answer_markdown(validate("Topics:\n- Web Services [2]\n- PHP and Node.js [2]"))
    assert "\n- <span" in md and md.count("<span") == 1


def test_answer_html_and_images_from_the_llm_are_not_rendered():
    a = answer('Pass marks <img src=x onerror="alert(1)"> are 45 [1]. ![logo](https://evil.example/t.png) $5 and $6')
    md = ui.answer_markdown(a)
    assert "<img" not in md and "&lt;img" in md
    assert "evil.example" not in md and "logo" in md
    assert r"\$5 and \$6" in md


ATTENTION = (
    "The formula for Scaled Dot-Product Attention [1] is:\n\n"
    r"$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$"
    "\n\nWhere:\n\n- $Q$, $K$, and $V$ are the packed queries, keys and values [2].\n"
    "- $d_k$ is the dimension of the queries and keys [2]."
)


def test_formulas_are_rendered_as_math():
    md = ui.answer_markdown(answer(ATTENTION))
    # A $$…$$ line becomes a display block; inline $…$ is kept; markers still become superscripts.
    block = ("is:\n\n$$\n" r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
             "\n$$\n\nWhere:")
    assert block in md
    assert "- $Q$, $K$, and $V$ are" in md and "- $d_k$ is" in md
    assert r"\$" not in md and "<sup>[1]</sup>" in md and "<sup>[2]</sup>" in md
    assert block in ui.streaming_markdown(ATTENTION)


def test_dollar_amounts_are_not_math():
    for text in ("$5 and $10", "between $5-$10", "costs $5, or $50 a year", r"pay \$5 or \$6"):
        md = ui.answer_markdown(answer(text))
        assert md.count(r"\$") == 2 and md.count("$") == 2, md
    assert ui.answer_markdown(answer("$5 and $x$")) == r"\$5 and $x$"


def test_display_math_in_running_text_or_a_list_item():
    md = ui.answer_markdown(answer(r"It is $$\frac{a}{b}$$ here [1]."))
    assert md == r"It is $\displaystyle \frac{a}{b}$ here <sup>[1]</sup>."
    md = ui.answer_markdown(answer("- Formula:\n\n  $$\n  a^2 +\n  b^2\n  $$\n- next"))
    assert "\n\n  $$\n  a^2 +\n  b^2\n  $$\n- next" in md  # stays inside the list item


def test_formulas_cannot_carry_html_or_markers():
    md = ui.answer_markdown(answer('$a <img src=x onerror="alert(1)"> b [1]$'))
    assert "<" not in md and r"$a \lt img src=x onerror=" in md and "b [1]$" in md


def test_streaming_shows_a_formula_once_it_is_closed():
    assert ui.streaming_markdown("where $d_k") == r"where \$d_k"
    assert ui.streaming_markdown("where $d_k$ is") == "where $d_k$ is"
    assert ui.streaming_markdown("$$x") == r"\$\$x"


def test_highlight_never_cuts_a_formula():
    a = answer("So $a = 1. b = 2$ holds.")
    plain = "".join(s.text for s in a.segments)
    start = plain.index("b = 2")  # a statement starting inside the formula
    a.validation = Validation([Check("b = 2 holds.", [], "warn", ["unsupported"], start=start, end=len(plain))])
    md = ui.answer_markdown(a)
    assert md.startswith(f'So <span style="{ui.WARN_STYLE}" title="unsupported">$a = 1. b = 2$ holds.</span>')


def test_passage_html_highlights_the_quote_on_one_line():
    html = ui.passage_html("Line one.\n\n<b>Pass Marks: 45.</b>", 11, 33)
    assert "\n" not in html and "<b>" not in html
    assert '">&lt;b&gt;Pass Marks: 45.&lt;/b&gt;</mark>' in html


def test_check_lines_and_summary():
    a = validate("The pass marks are 45 [1]. Unit VII covers PHP [2].")
    lines = ui.check_lines(a.validation)
    assert lines[0].startswith("✓ The pass marks are 45") and "support 0.95" in lines[0]
    assert lines[1].startswith(":orange[✗] Unit VII covers PHP") and "PHP" in lines[1]
    assert ui.summary_line(a) == ("1/2 statements supported by their citations, 1 warning. "
                                  "Unsupported statements are highlighted.")
    assert ui.summary_line(answer("The pass marks are 45 [1].")) is None  # not validated


def test_escape_md_shows_text_literally():
    assert ui.escape_md("**a** [1] $5 :red[x] <b>") == r"\*\*a\*\* \[1\] \$5 \:red\[x\] &lt;b&gt;"


# -- documents ---------------------------------------------------------------------------


def test_save_upload_keeps_only_the_file_name(tmp_path):
    path = ui.save_upload("../../evil.pdf", b"%PDF", tmp_path / "data")
    assert path == tmp_path / "data" / "evil.pdf" and path.read_bytes() == b"%PDF"
    assert ui.save_upload("C:\\docs\\notes.md", b"x", tmp_path).name == "notes.md"
    for bad in ("run.sh", ".env", "../"):
        with pytest.raises(ValueError):
            ui.save_upload(bad, b"x", tmp_path)


def test_source_pdf_stays_inside_the_folder(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "secret.pdf").write_bytes(b"%PDF")
    (tmp_path / "data" / "notes.md").write_text("x")
    assert ui.source_pdf("a.pdf", tmp_path / "data") == (tmp_path / "data" / "a.pdf").resolve()
    (tmp_path / "data" / "scan.jpg").write_bytes(b"jpg")
    assert ui.source_image("scan.jpg", tmp_path / "data") == (tmp_path / "data" / "scan.jpg").resolve()
    assert ui.source_image("a.pdf", tmp_path / "data") is None and ui.source_pdf("scan.jpg", tmp_path / "data") is None
    for source in ("../secret.pdf", "notes.md", "missing.pdf"):
        assert ui.source_pdf(source, tmp_path / "data") is None


def test_page_pdf_cuts_out_the_cited_pages(tmp_path):
    writer = PdfWriter()
    for width in (100, 200, 300):
        writer.add_blank_page(width=width, height=100)
    writer.write(tmp_path / "doc.pdf")
    reader = PdfReader(__import__("io").BytesIO(ui.page_pdf(tmp_path / "doc.pdf", 2, 3)))
    assert [p.mediabox.width for p in reader.pages] == [200, 300]


def test_report_text():
    r = IndexReport(added=["a.pdf"], unchanged=["README.md"], chunks=4)
    assert ui.report_text(r) == "Added: a.pdf; Unchanged: README.md; 4 chunks embedded"
    assert ui.report_text(IndexReport()) == "Nothing to index"


def test_has_key_ignores_placeholders(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "real")
    assert not ui.has_key("anthropic") and ui.has_key("gemini")


# -- the app (headless, with fakes) --------------------------------------------------------


class FakeStore:
    docs = {"syllabus.pdf": 2}

    def __init__(self):
        self.embedder = NS(model="EMBEDDING MODEL")
        self.removed = []

    def sources(self):
        return dict(self.docs)

    def count(self):
        return sum(self.docs.values())

    def delete_source(self, source):
        self.removed.append(source)
        return 1


class FakeRetriever:
    reranker = None
    calls: list = []  # the `sources` of every search, to check what a user could search

    def __init__(self, store):
        self.store = store

    def retrieve(self, question, k, sources):
        FakeRetriever.calls.append(sources)
        return HITS  # whatever was asked: a retriever that ignores the filter


class FailingBackend:
    model = "fake"

    def generate(self, question, hits, on_text=None):
        raise GenerationError("Free-tier limit reached. Wait a minute and retry.")


class ChatBackend(FakeBackend):
    """Answers like FakeBackend; rewrites any follow-up to `rewrite`."""

    rewrite = "What are the full marks for Web Technology?"

    def __init__(self, text):
        super().__init__(text)
        self.completions = []

    def complete(self, system, prompt):
        self.completions.append(prompt)
        return self.rewrite


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHATS_DIR", tmp_path / "chats")
    monkeypatch.setattr(FakeRetriever, "calls", [])
    monkeypatch.setattr("citerag.auth.configured", lambda: False)  # local mode, whatever secrets.toml says

    def run(reply="The pass marks are 45 [1]. Unit VII covers PHP [2].", docs=None, backend=None,
            access=None, email=None):
        if access is not None:  # sign-in on: `email` is who Google says is signed in (None: nobody)
            (tmp_path / "access.toml").write_text(access)
            monkeypatch.setattr(config, "ACCESS_FILE", tmp_path / "access.toml")
            monkeypatch.setattr("citerag.auth.configured", lambda: True)
            monkeypatch.setattr("citerag.auth.identity", lambda: email)
        monkeypatch.setattr(FakeStore, "docs", {"syllabus.pdf": 2} if docs is None else docs)
        monkeypatch.setattr("citerag.store.Store", FakeStore)
        monkeypatch.setattr("citerag.retriever.Retriever", FakeRetriever)
        nli = FakeNLI()
        nli.model = "NLI MODEL"
        monkeypatch.setattr("citerag.validator.Validator", lambda: Validator(nli, threshold=0.5))
        monkeypatch.setattr("citerag.generator.make_backend", lambda provider: backend or ChatBackend(reply))
        st.cache_resource.clear()
        return AppTest.from_file(APP, default_timeout=30).run()

    yield run
    st.cache_resource.clear()


def test_empty_index_asks_for_documents(app):
    at = app(docs={})
    assert "Add a PDF" in at.info[0].value and at.chat_input[0].disabled


def test_loading_the_models_draws_nothing_on_the_page(app, monkeypatch):
    monkeypatch.setattr(FakeRetriever, "reranker", NS(model="RERANK MODEL"))
    at = app()  # Streamlit "magic" draws bare expressions, e.g. a model preload
    assert not any("MODEL" in m.value for m in at.markdown)


def test_answer_shows_highlights_sources_and_checks(app):
    at = app().chat_input[0].set_value("What are the pass marks?").run()
    assert not at.exception
    body = [m.value for m in at.markdown]
    assert "**What are the pass marks?**" in body
    answer_md = next(m for m in body if "<sup>[1]</sup>" in m)
    assert "<span" in answer_md and "PHP" in answer_md
    labels = [e.label for e in at.expander]
    assert labels[:2] == [r"\[1\] syllabus\.pdf, p. 1", r"\[2\] syllabus\.pdf, p. 3"]
    assert "Checks" in labels and "Retrieved passages" in labels
    assert at.main.caption[-1].value.startswith("1/2 statements supported")


def test_strict_mode_shows_only_supported_statements(app):
    at = app()
    at.sidebar.toggle[1].set_value(True)
    at = at.chat_input[0].set_value("What are the pass marks?").run()
    answer_md = next(m.value for m in at.markdown if "<sup>" in m.value)
    assert answer_md == "The pass marks are 45 <sup>[1]</sup>."
    assert "strict mode removed 1 unsupported statement" in at.main.caption[-1].value


def test_backend_errors_are_shown_plainly(app):
    at = app(backend=FailingBackend()).chat_input[0].set_value("q").run()
    assert not at.exception and at.error[0].value == "Free-tier limit reached. Wait a minute and retry."


def test_remove_button_drops_a_document_from_the_index(app):
    at = app()
    at.sidebar.button[0].click().run()
    assert at.success[0].value == "Removed syllabus.pdf from the index."


# -- chats (Phase 12) ---------------------------------------------------------------


def test_a_follow_up_is_searched_as_a_standalone_question_and_the_chat_is_saved(app, tmp_path):
    backend = ChatBackend("The pass marks are 45 [1].")
    at = app(backend=backend)
    at = at.chat_input[0].set_value("What are the pass marks?").run()
    at = at.chat_input[0].set_value("And the full marks?").run()
    assert not at.exception
    captions = [c.value for c in at.main.caption]
    assert "Searched for: What are the full marks for Web Technology?" in captions
    assert len(backend.completions) == 1  # only the follow-up was rewritten
    chat_id = at.query_params["chat"][0] if isinstance(at.query_params["chat"], list) else at.query_params["chat"]
    saved = ChatStore(tmp_path / "chats").load(chat_id)
    assert [(t.question, t.search) for t in saved.turns] == [
        ("What are the pass marks?", "What are the pass marks?"),
        ("And the full marks?", "What are the full marks for Web Technology?")]


def test_a_saved_chat_reopens_from_the_address(app, tmp_path):
    store = ChatStore(tmp_path / "chats")
    chat = Chat()
    chat.turns.append(Turn("What are the pass marks?", "What are the pass marks?", validate("The pass marks are 45 [1].")))
    store.save(chat)
    at = app()
    assert "**What are the pass marks?**" not in [m.value for m in at.markdown]  # no address: a new chat
    reopened = AppTest.from_file(APP, default_timeout=30)  # a fresh browser session, e.g. after a reload
    reopened.query_params["chat"] = chat.id
    reopened = reopened.run()
    assert "**What are the pass marks?**" in [m.value for m in reopened.markdown]
    assert any(b.label == "Delete this chat" for b in reopened.sidebar.button)
    assert any(b.label == "What are the pass marks?" for b in reopened.sidebar.button)  # listed under Chats


# -- sign-in and permissions (Phase 13) ---------------------------------------------

ACCESS = """
[[users]]
email = "boss@example.com"
admin = true

[[users]]
email = "asha@example.com"
groups = ["hr"]
"""
GROUP_DOCS = {"hr/policy.pdf": 2, "finance/budget.xlsx": 1, "public/guide.pdf": 1, "loose.pdf": 1}


def sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown)


def test_nobody_signed_in_sees_only_the_sign_in_button(app):
    at = app(docs=GROUP_DOCS, access=ACCESS, email=None)
    assert [b.label for b in at.button] == ["Sign in with Google"]
    assert not at.chat_input and not at.sidebar.markdown and FakeRetriever.calls == []


def test_an_unlisted_account_is_turned_away(app):
    at = app(docs=GROUP_DOCS, access=ACCESS, email="stranger@example.com")
    assert "isn't on the access list" in at.error[0].value
    assert not at.chat_input and [b.label for b in at.button] == ["Sign out"]


def test_a_member_sees_and_searches_only_their_groups(app):
    at = app(docs=GROUP_DOCS, access=ACCESS, email="asha@example.com")
    shown = sidebar_text(at)
    assert "hr/policy\\.pdf" in shown or "hr/policy" in shown
    assert "guide" in shown and "budget" not in shown and "loose" not in shown
    labels = [b.label for b in at.sidebar.button]
    assert "Sign out" in labels and not any("Rescan" in l for l in labels)  # no document management
    assert not at.sidebar.selectbox and not at.get("file_uploader")
    at = at.chat_input[0].set_value("What are the pass marks?").run()
    assert not at.exception
    assert FakeRetriever.calls == [["hr/policy.pdf", "public/guide.pdf"]]  # never asked for more
    # The fake retriever answered with syllabus.pdf anyway: dropped, so nothing reaches the model or page.
    assert not any("syllabus" in e.label for e in at.expander)
    assert "45" not in " ".join(m.value for m in at.main.markdown if "pass marks" not in m.value)


def test_an_admin_sees_everything_and_chooses_the_folder_for_uploads(app):
    at = app(docs=GROUP_DOCS, access=ACCESS, email="boss@example.com")
    shown = sidebar_text(at)
    assert all(word in shown for word in ("policy", "budget", "guide", "loose"))
    [folder] = at.sidebar.selectbox
    assert folder.label == "Into folder (group)" and folder.value == "public"
    assert any("Rescan" in b.label for b in at.sidebar.button)


def test_chats_are_kept_per_user(app, tmp_path):
    at = app(docs=GROUP_DOCS, access=ACCESS, email="boss@example.com")
    at.chat_input[0].set_value("What are the pass marks?").run()
    saved = [p.parent.name for p in (tmp_path / "chats").rglob("*.json")]
    assert len(saved) == 1 and saved[0] != "chats"  # in the user's own folder
    other = app(docs=GROUP_DOCS, access=ACCESS, email="asha@example.com")
    assert not any(b.key and b.key.startswith("chat-") for b in other.sidebar.button)


def test_a_broken_access_list_is_reported(app):
    at = app(docs=GROUP_DOCS, access='[[users]]\nemail = "oops"', email="boss@example.com")
    assert "access list has a problem" in at.error[0].value and not at.chat_input


def test_broken_sign_in_settings_stop_the_app_instead_of_opening_it(app, monkeypatch):
    def broken():
        raise auth.AuthConfigError(".streamlit/secrets.toml can't be read (Illegal character)")

    monkeypatch.setattr("citerag.auth.configured", broken)
    at = app(docs=GROUP_DOCS)  # before the fix: local mode, everything readable as admin
    assert not at.exception
    assert "sign-in settings have a problem" in at.error[0].value and "Illegal character" in at.error[0].value
    assert not at.chat_input and not at.sidebar.markdown and not at.button and FakeRetriever.calls == []
