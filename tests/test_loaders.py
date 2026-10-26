import logging

import pytest
from pypdf import PdfReader, PdfWriter

from citerag.chunker import Chunker
from citerag.loaders import (
    LoaderError,
    Page,
    _infer_paragraphs,
    clean_pdf_pages,
    clean_text,
    load_dir,
    load_file,
    load_paths,
    load_pdf,
    load_text,
)


def make_pdf(path, pages):
    """Write a minimal PDF; `pages` is a list of pages, each a list of text lines."""
    font_id = 3 + 2 * len(pages)
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages))), len(pages)
        ),
        font_id: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, lines in enumerate(pages):
        ops = []
        for j, line in enumerate(lines):
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"BT /F1 10 Tf 50 {780 - 14 * j} Td ({esc}) Tj ET")
        stream = "\n".join(ops)
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        objects[content_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    out, offsets = b"%PDF-1.4\n", {}
    for n in sorted(objects):
        offsets[n] = len(out)
        out += f"{n} 0 obj\n{objects[n]}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offsets[n]:010d} 00000 n \n" for n in sorted(objects)).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


TOPICS = ["apples", "bridges", "comets", "deserts", "engines"]


def body(page_no):
    # Distinct words per page: lines that differ only in digits look like a running header.
    topic = TOPICS[page_no - 1]
    return [
        f"This page is about {topic} and its body text runs across",
        f"several lines, so the loader sees a realistic width for {topic}.",
        f"It ends with a short line on {topic}.",
    ]


# -- text files ----------------------------------------------------------------


def test_load_text_is_single_page_named_after_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nSome text.\n", encoding="utf-8")
    assert load_text(f) == [Page("# Title\n\nSome text.", "notes.md", 1)]


def test_load_text_falls_back_to_cp1252(tmp_path, caplog):
    f = tmp_path / "old.txt"
    f.write_bytes("Caf\xe9 \x93quoted\x94".encode("latin-1"))
    with caplog.at_level(logging.WARNING):
        [page] = load_text(f)
    assert page.text == "Café “quoted”"
    assert "not valid UTF-8" in caplog.text


def test_empty_text_file_is_skipped(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("  \n\n ", encoding="utf-8")
    assert load_text(f) == []


def test_clean_text_normalizes_characters_and_keeps_paragraphs():
    raw = "\ufeffThe \ufb01rst\u00a0line   \r\nnext\u200b line\r\n\r\n\r\n\r\nNew para\x00graph\u2028end"
    assert clean_text(raw) == "The first line\nnext line\n\nNew paragraph\nend"


# -- PDFs ----------------------------------------------------------------------


def test_load_pdf_one_page_record_per_page(tmp_path):
    pdf = make_pdf(tmp_path / "report.pdf", [body(1), body(2)])
    pages = load_pdf(pdf)
    assert [(p.source, p.page) for p in pages] == [("report.pdf", 1), ("report.pdf", 2)]
    assert "bridges" in pages[1].text


def test_blank_pages_are_skipped_but_page_numbers_stay_true(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "gaps.pdf", [body(1), [], body(3)])
    with caplog.at_level(logging.WARNING):
        pages = load_pdf(pdf)
    assert [p.page for p in pages] == [1, 3]
    assert "skipped 1 page(s) with no text: 2" in caplog.text


def test_pdf_without_text_warns_scanned(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "scan.pdf", [[], []])
    with caplog.at_level(logging.WARNING):
        assert load_pdf(pdf) == []
    assert "probably scanned" in caplog.text


def test_running_headers_footers_and_page_numbers_removed(tmp_path):
    pages = [
        ["ACME Annual Report 2024", *body(i), f"Page {i} of 4"] for i in range(1, 5)
    ]
    texts = [p.text for p in load_pdf(make_pdf(tmp_path / "r.pdf", pages))]
    assert len(texts) == 4
    for t in texts:
        assert "ACME" not in t and "of 4" not in t
        assert t.startswith("This page is about")


def test_headers_kept_when_document_too_short_to_tell():
    texts = clean_pdf_pages(["Intro\nBody one.", "Intro\nBody two."])
    assert all(t.startswith("Intro") for t in texts)


def test_bare_page_numbers_removed_only_at_page_edges():
    [text] = clean_pdf_pages(["12\nItems:\n42\nmore items\nend.\nxiv"])
    assert text.split() == ["Items:", "42", "more", "items", "end."]


def test_symbol_font_bullets_become_bullets():
    assert clean_text(chr(0xF0B7) + " Validate forms\n" + chr(0xF0A7) + " Use DOM") == (
        "• Validate forms\n• Use DOM"
    )


def test_line_ending_mid_sentence_is_never_split_off():
    # Real table cells from a syllabus PDF: "HTML5" / "XHTML" were becoming
    # standalone paragraphs, which the chunker then took for ALL-CAPS headings.
    lines = [
        "2.5. HTML Layouts: Using Tables, DIV & Span Tags, HTML Style Sheets,",
        "HTML Entities, Events and many other things that fill a whole line.",
        "• Understand XHTML and",
        "HTML5",
        "• Differentiate HTML from",
        "XHTML",
        "Unit III: XHTML & HTML5 (4 hr)",
    ]
    paragraphs = _infer_paragraphs(lines).split("\n\n")
    assert "• Understand XHTML and\nHTML5" in paragraphs
    assert "• Differentiate HTML from\nXHTML" in paragraphs
    assert "HTML5" not in paragraphs and "XHTML" not in paragraphs


def test_hyphenated_line_breaks_rejoined():
    [text] = clean_pdf_pages(["A long line about informa-\ntion retrieval systems."])
    assert "information retrieval" in text


def test_paragraphs_and_headings_inferred_from_line_shapes():
    lines = [
        "2. Methods",
        "We collected documents from many places and cleaned them before",
        "indexing them into the store for later retrieval.",
        "Then we evaluated the system on a set of questions written by the",
        "team, measuring how often answers cited the right page.",
    ]
    assert _infer_paragraphs(lines).split("\n\n") == [
        "2. Methods",
        "\n".join(lines[1:3]),
        "\n".join(lines[3:5]),
    ]


def test_password_protected_pdf_raises(tmp_path):
    plain = make_pdf(tmp_path / "plain.pdf", [body(1)])
    writer = PdfWriter(clone_from=PdfReader(plain))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    with open(locked, "wb") as fh:
        writer.write(fh)
    with pytest.raises(LoaderError, match="password"):
        load_pdf(locked)


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(LoaderError, match="cannot read PDF"):
        load_pdf(bad)


# -- folders and dispatch --------------------------------------------------------


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "sheet.xlsx"
    f.write_bytes(b"x")
    with pytest.raises(LoaderError, match="unsupported"):
        load_file(f)


def test_load_dir_relative_sources_sorted_skips_bad_and_hidden(tmp_path, caplog):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "b.txt").write_text("Bee text.", encoding="utf-8")
    (tmp_path / "sub" / "a.md").write_text("Ay text.", encoding="utf-8")
    (tmp_path / ".hidden" / "x.txt").write_text("Hidden.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "broken.pdf").write_bytes(b"garbage")
    make_pdf(tmp_path / "c.pdf", [body(1)])
    with caplog.at_level(logging.WARNING):
        pages = load_dir(tmp_path)
    assert [p.source for p in pages] == ["b.txt", "c.pdf", "sub/a.md"]
    assert "broken.pdf" in caplog.text


def test_load_paths_mixes_files_and_folders(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("Folder file.", encoding="utf-8")
    single = tmp_path / "single.txt"
    single.write_text("Single file.", encoding="utf-8")
    assert [p.source for p in load_paths([single, tmp_path / "docs"])] == ["single.txt", "a.txt"]


def test_loaded_pdf_feeds_chunker_with_page_spans(tmp_path):
    pdf = make_pdf(tmp_path / "guide.pdf", [body(1), body(2), body(3)])
    chunker = Chunker(40, 10, 8, count_tokens=lambda ts: [len(t.split()) for t in ts])
    chunks = chunker.chunk_pages(load_pdf(pdf))
    assert chunks and all(c.source == "guide.pdf" for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
