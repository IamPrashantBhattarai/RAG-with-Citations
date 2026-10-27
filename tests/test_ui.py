from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import streamlit as st
from pypdf import PdfReader, PdfWriter
from streamlit.testing.v1 import AppTest

from citerag import ui
from citerag.generator import GenerationError
from citerag.store import IndexReport
from citerag.validator import Validator
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

    def __init__(self, store):
        self.store = store

    def retrieve(self, question, k, sources):
        return HITS


class FailingBackend:
    model = "fake"

    def generate(self, question, hits, on_text=None):
        raise GenerationError("Free-tier limit reached. Wait a minute and retry.")


@pytest.fixture
def app(monkeypatch):
    def run(reply="The pass marks are 45 [1]. Unit VII covers PHP [2].", docs=None, backend=None):
        monkeypatch.setattr(FakeStore, "docs", {"syllabus.pdf": 2} if docs is None else docs)
        monkeypatch.setattr("citerag.store.Store", FakeStore)
        monkeypatch.setattr("citerag.retriever.Retriever", FakeRetriever)
        nli = FakeNLI()
        nli.model = "NLI MODEL"
        monkeypatch.setattr("citerag.validator.Validator", lambda: Validator(nli, threshold=0.5))
        monkeypatch.setattr("citerag.generator.make_backend", lambda provider: backend or FakeBackend(reply))
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
