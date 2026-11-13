from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("docling_core")

from docling_core.types.doc import (BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size, TableCell,
                                    TableData)

from citerag import config, layout
from tests.test_loaders import body, make_pdf

DATA = Path(__file__).parent.parent / "data"
# The scanned test form, wherever it is under data/ (data/hr/ since the Phase 13 folders).
FORM = next(iter(sorted(DATA.rglob("handwriting.pdf"))), DATA / "handwriting.pdf")


@pytest.fixture(autouse=True)
def parse_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PARSE_CACHE_DIR", tmp_path / "parsed")
    return tmp_path / "parsed"


def prov(page):
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0))


def table(rows, header_rows=1, spans=()):
    """A TableItem's data from rows of cell texts; `spans` = (row, col, width) merged cells."""
    cells, covered = [], set()
    for r, c, w in spans:
        covered |= {(r, c + i) for i in range(1, w)}
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if (r, c) in covered:
                continue
            w = next((w for sr, sc, w in spans if (sr, sc) == (r, c)), 1)
            cells.append(TableCell(text=text, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                                   start_col_offset_idx=c, end_col_offset_idx=c + w, col_span=w,
                                   column_header=r < header_rows))
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


def doc_with(*items):
    doc = DoclingDocument(name="t")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=600, height=800))
    for kind, page, *args in items:
        if kind == "heading":
            doc.add_heading(args[0], prov=prov(page))
        elif kind == "text":
            doc.add_text(DocItemLabel.TEXT, args[0], prov=prov(page))
        elif kind == "formula":
            doc.add_formula(args[0], orig=args[1], prov=prov(page))
        elif kind == "table":
            caption = doc.add_text(DocItemLabel.CAPTION, args[1], prov=prov(page))
            doc.add_table(data=args[0], caption=caption, prov=prov(page))
    return doc


# -- tables ----------------------------------------------------------------------------


def test_table_rows_name_their_columns():
    doc = doc_with(("table", 1, table([
        ["", "N", "dmodel", "BLEU (dev)", "params ×106"],
        ["base", "6", "512", "25.8", "65"],
        ["big", "6", "1024", "26.4", "213"],
        ["", "", "256", "24.5", ""],  # blank cells mean "same as base": left out
    ]), "Table 3: Variations"))
    [item] = doc.tables
    assert layout.table_rows(item) == (
        "- base; N: 6; dmodel: 512; BLEU (dev): 25.8; params ×106: 65\n"
        "- big; N: 6; dmodel: 1024; BLEU (dev): 26.4; params ×106: 213\n"
        "- dmodel: 256; BLEU (dev): 24.5"
    )


def test_stacked_headers_join_and_merged_cells_are_written_once():
    data = table([["Model", "BLEU", "BLEU"], ["", "EN-DE", "EN-FR"], ["Transformer (big)", "28.4", "41.0"],
                  ["Note", "same for both", "same for both"]],
                 header_rows=2, spans=[(0, 1, 2), (3, 1, 2)])
    [item] = doc_with(("table", 1, data, "Table 2")).tables
    assert layout.table_rows(item) == ("- Model: Transformer (big); BLEU EN-DE: 28.4; BLEU EN-FR: 41.0\n"
                                       "- Model: Note; BLEU EN-DE: same for both")


def test_table_without_header_flags_uses_its_first_row():
    [item] = doc_with(("table", 1, table([["Name", "Width"], ["G/14", "1664"]], header_rows=0), "T")).tables
    assert layout.table_rows(item) == "- Name: G/14; Width: 1664"


# -- formulas --------------------------------------------------------------------------

# What Docling's formula model wrote for the Attention paper, and the PDF's own text.
ATTENTION = (r"A t t e n t ( Q , K , V ) = \text {softmax} ( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } ) V & & ( 1 )",
             "Attention(Q, K, V ) = softmax( QKT √dk )V (1)")
FFN = (r"F F N ( x ) = \max ( 0 , x W _ { 1 } + b _ { 1 } ) W _ { 2 } + b _ { 2 }",
       "FFN(x) = max(0, xW1 + b1)W2 + b2 (2)")
LRATE = (r"l r a t e = d _ { \text {mod} } ^ { - 0 . 5 } \cdot \min ( s t e p",
         "lrate = d-0.5 model · min(step")
PE = (r"P E _ { ( p o s , 2 i ) } & = s i n ( p o s / 1 0 0 0 0 ^ { 2 i / d _ { \text {model} } } ) \\ "
      r"P E _ { ( p o s , 2 i + 1 ) } & = \cos ( p o s / 1 0 0 0 0 ^ { 2 i / d _ { \text {model} } } )",
      "PE(pos,2i) = sin(pos/100002i/dmodel ) PE(pos,2i+1) = cos(pos/100002i/dmodel )")


def test_latex_is_kept_only_when_it_matches_the_pdf_text():
    assert layout.formula_text(*FFN) == r"$$FFN(x)=\max(0,xW_{1}+b_{1})W_{2}+b_{2}$$"
    assert layout.formula_text(*PE).startswith(r"$$\begin{aligned}PE_{(pos,2i)}&=sin(")  # two lines, kept
    assert layout.formula_text(*ATTENTION) == ATTENTION[1]  # "Attent": misread, PDF text kept
    assert layout.formula_text(*LRATE) == LRATE[1]  # "mod" for "model"
    assert layout.formula_text("", "E = mc2") == "E = mc2"  # no LaTeX (the default mode)
    assert layout.formula_text("x ^ { 2 }", "x2", inline=True) == "$x^{2}$"


def test_clean_latex():
    assert layout.clean_latex(ATTENTION[0]) == r"Attent(Q,K,V)=\text{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V"
    assert layout.clean_latex(r"a \cdot b") == r"a\cdot b"  # a command keeps the space before a letter
    assert layout.clean_latex(r"\text { head 1 }") == r"\text{head 1}"  # words inside \text keep spaces
    assert layout.clean_latex(r"a & = b \\ c & = d \\") == r"\begin{aligned}a&=b\\c&=d\end{aligned}"


def test_same_symbols_ignores_layout_not_letters():
    assert layout.same_symbols(r"\frac{QK^{T}}{\sqrt{d_{k}}}", "QKT √dk")
    assert layout.same_symbols(r"\sin(pos)", "sin(pos)")  # \sin prints its name
    assert not layout.same_symbols(r"Attent(Q)", "Attention(Q)")
    assert not layout.same_symbols(r"\alpha", "")  # nothing to compare with


# -- pages -----------------------------------------------------------------------------


def test_page_texts_one_markdown_text_per_page():
    doc = doc_with(
        ("heading", 1, "3.2 Attention"),
        ("text", 1, "We compute the matrix of outputs as:"),
        ("formula", 1, "", ATTENTION[1]),
        ("table", 2, table([["Model", "BLEU"], ["big", "28.4"]]), "Table 2: Results."),
    )
    p1, p2 = layout.page_texts(doc)
    assert p1 == f"## 3.2 Attention\n\nWe compute the matrix of outputs as:\n\n{ATTENTION[1]}"
    assert p2 == "Table 2: Results.\n\n- Model: big; BLEU: 28.4"


def test_parse_file_is_cached_by_content_and_settings(tmp_path, monkeypatch, parse_cache):
    calls = []

    class FakeConverter:
        def convert(self, path):
            calls.append(path)
            return NS(document=doc_with(("text", 1, "Hello."), ("text", 2, "World.")), pages=[])

    monkeypatch.setattr(layout, "_converter", lambda formulas, ocr: FakeConverter())
    monkeypatch.setattr(layout, "text_layer_chars", lambda path: [100, 100])  # no page needs OCR
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF one")
    assert layout.parse_file(pdf) == ["Hello.", "World."]
    assert layout.parse_file(pdf) == ["Hello.", "World."] and len(calls) == 1
    layout.parse_file(pdf, formulas="latex")  # different settings: parsed again
    layout.parse_file(pdf, ocr="off")
    pdf.write_bytes(b"%PDF two")  # changed file: parsed again
    layout.parse_file(pdf)
    assert len(calls) == 4 and len(list(parse_cache.iterdir())) == 4


def test_conversion_errors_become_layout_errors(tmp_path, monkeypatch):
    class Broken:
        def convert(self, path):
            raise RuntimeError("bad page tree")

    monkeypatch.setattr(layout, "_converter", lambda formulas, ocr: Broken())
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    with pytest.raises(layout.LayoutError, match="bad page tree"):
        layout.parse_file(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="CITERAG_PDF_FORMULAS"):
        layout.parse_file(tmp_path / "x.pdf", formulas="mathml")
    with pytest.raises(ValueError, match="CITERAG_OCR"):
        layout.parse_file(tmp_path / "x.pdf", ocr="tesseract")
    monkeypatch.setattr(layout.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="pip install ocrmac"):
        layout.parse_file(tmp_path / "x.pdf", ocr="ocrmac")


# -- forms -----------------------------------------------------------------------------


def test_rows_text_reads_each_visual_row_left_to_right():
    boxes = [  # (top, bottom, left, right, text), in no particular order
        (500, 512, 400, 460, "$60000"), (500, 512, 40, 200, "6a Total plan assets"),
        (501, 513, 300, 360, "$50000"),
        (520, 532, 40, 200, "b  Total plan liabilities"), (521, 535, 300, 350, "$4000"),
        (10, 30, 300, 350, "Form"), (11, 30, 358, 440, "Number:"), (10, 31, 450, 560, "CA530082"),  # one phrase
        (600, 612, 40, 60, "  "),  # empty boxes are dropped
    ]
    assert layout.rows_text(boxes) == ("Form Number: CA530082\n"
                                       "6a Total plan assets   $50000   $60000\n"
                                       "b Total plan liabilities   $4000")


def page(form_share, cells, page_no=1, checkbox=None):
    """A fake Docling page: one 'form' region covering `form_share` of a 100x100 page."""
    def box(top, left, right):
        b = NS(t=top, b=top + 5, l=left, r=right)
        return NS(to_top_left_origin=lambda page_height: b, area=lambda: 25.0)

    def cell(i, top, left, text):
        return NS(index=i, text=text, rect=NS(to_bounding_box=lambda: box(top, left, left + 4 * len(text))))

    form = NS(label=NS(value="form"), bbox=NS(area=lambda: form_share * 100 * 100),
              cells=[cell(i, *c) for i, c in enumerate(cells)])
    header = NS(label=NS(value="page_header"), bbox=NS(area=lambda: 50.0), cells=[cell(99, 1, 10, "Form No. 7")])
    clusters = [header, form]
    if checkbox:
        clusters.append(NS(label=NS(value=checkbox[0]), bbox=box(*checkbox[1:]), cells=[]))
    return NS(page_no=page_no, size=NS(width=100, height=100), predictions=NS(layout=NS(clusters=clusters)))


def test_only_pages_that_are_mostly_a_form_are_written_by_rows():
    result = NS(pages=[page(0.76, [(50, 10, "Total assets"), (50, 80, "50000"), (70, 10, "(1) first return")],
                            checkbox=("checkbox_selected", 70, 2, 6)),
                       page(0.20, [(50, 10, "a small form box")], page_no=2)])
    assert layout.form_pages(result) == {1: "Form No. 7\nTotal assets   50000\n[x] (1) first return"}
    # A checkbox drawn inside a line's box ("(1) [x] the first…") still goes in front of it.
    inside = NS(pages=[page(0.9, [(70, 10, "(1) the first return")], checkbox=("checkbox_selected", 70, 22, 26))])
    assert layout.form_pages(inside)[1].endswith("[x] (1) the first return")


@pytest.mark.skipif(not (layout.available() and FORM.is_file()), reason="docling or handwriting.pdf (under data/) missing")
def test_handwritten_form_values_stay_next_to_their_labels():
    [text] = layout.parse_file(FORM, ocr="rapidocr")
    lines = text.split("\n")
    assert lines[0] == "Form Number: CA530082"
    assert any(l.startswith("6a Total plan assets") and "50000" in l and "60000" in l for l in lines)
    assert any("5a(1)" in l and l.endswith("10") for l in lines)
    for value in ("Acme Corp", "735268329", "011536259", "02/05/2022", "532678", "5732900"):
        assert value in text


# -- OCR -------------------------------------------------------------------------------


def draw(path, lines, size=(1200, 500)):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", size, 255)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(lines):
        ImageDraw.Draw(img).text((60, 60 + 70 * i), line, fill=0, font=font)
    img.save(path)
    return img


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_ocr_reads_images_and_scanned_pdfs(tmp_path):
    lines = ["Invoice number 4471", "Total due: 980 rupees", "Paid by bank transfer"]
    img = draw(tmp_path / "invoice.png", lines)
    img.save(tmp_path / "scan.pdf", "PDF", resolution=150)  # an image-only PDF, like a scanner makes
    for path in (tmp_path / "invoice.png", tmp_path / "scan.pdf"):
        [text] = layout.parse_file(path, ocr="rapidocr")
        for word in ("Invoice", "4471", "980", "transfer"):
            assert word in text, (path.name, text)
    assert layout.parse_file(tmp_path / "scan.pdf", ocr="off") == [""]  # no text layer, no OCR


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_only_pages_without_a_text_layer_are_ocrd(tmp_path):
    from pypdf import PdfReader, PdfWriter

    draw(tmp_path / "p.png", ["Scanned page about volcanoes"]).save(tmp_path / "scan.pdf", "PDF", resolution=150)
    text_pdf = make_pdf(tmp_path / "text.pdf", [body(1), body(2)])
    writer = PdfWriter()
    for part in (text_pdf, tmp_path / "scan.pdf"):
        for p in PdfReader(part).pages:
            writer.add_page(p)
    with open(tmp_path / "mixed.pdf", "wb") as fh:
        writer.write(fh)

    assert [n > 0 for n in layout.text_layer_chars(tmp_path / "mixed.pdf")] == [True, True, False]
    p1, p2, p3 = layout.parse_file(tmp_path / "mixed.pdf", ocr="rapidocr")
    assert [p1, p2] == layout.parse_file(text_pdf, ocr="off")  # text pages exactly as without OCR
    assert "volcanoes" in p3


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_real_docling_reads_a_pdf_page_by_page(tmp_path):
    pdf = make_pdf(tmp_path / "r.pdf", [body(1), [], body(3)])
    p1, p2, p3 = layout.parse_file(pdf)
    assert "apples" in p1 and p2 == "" and "comets" in p3
    assert "bridges" not in p1 + p3


# -- Word (Phase 11) -------------------------------------------------------------------

OMML = ('<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath>'
        '<m:sSup><m:e><m:r><m:t>a</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>'
        '<m:r><m:t>+</m:t></m:r>'
        '<m:sSup><m:e><m:r><m:t>b</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>'
        '</m:oMath></m:oMathPara>')


def make_docx(path):
    from docx import Document
    from docx.oxml import parse_xml

    doc = Document()
    doc.sections[0].header.paragraphs[0].text = "Northwind Analytics · Version 3.2"
    doc.sections[0].footer.paragraphs[0].text = "Document number HR-POL-014"
    doc.add_heading("Leave Policy", level=0)
    doc.add_heading("1. Annual leave", level=1)
    doc.add_paragraph("Employees receive 24 days of paid annual leave.")
    for step in ("Submit the request.", "Your manager approves it."):
        doc.add_paragraph(step, style="List Number")
    doc.add_heading("2. Travel", level=1)
    t = doc.add_table(rows=3, cols=3)
    t.cell(0, 0).merge(t.cell(1, 0)).text = "Grade"
    t.cell(0, 1).merge(t.cell(0, 2)).text = "Hotel limit (NPR)"
    t.cell(1, 1).text, t.cell(1, 2).text = "Kathmandu", "Other cities"
    for c, v in enumerate(("A (Director)", "18,000", "12,000")):
        t.cell(2, c).text = v
    doc.add_paragraph().add_run()._r.getparent().append(parse_xml(OMML))
    doc.save(path)
    return path


def test_latex_from_the_file_itself_is_kept():
    # Word stores equations as structure: Docling's LaTeX and its "original" are the same string.
    assert layout.formula_text(r"\frac{a}{b}", r"\frac{a}{b}") == r"$$\frac{a}{b}$$"
    assert layout.formula_text(r"\frac{a}{b}", "a/c") == "a/c"  # read from an image and disagreeing: not kept


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_word_file_is_one_text_with_sections_tables_and_header(tmp_path):
    [text] = layout.parse_file(make_docx(tmp_path / "policy.docx"))
    assert text.startswith("Northwind Analytics · Version 3.2\n\nDocument number HR-POL-014\n\n# Leave Policy")
    assert "## 1. Annual leave\n\nEmployees receive 24 days of paid annual leave." in text
    assert "1. Submit the request.\n2. Your manager approves it." in text
    assert "- Grade: A (Director); Hotel limit (NPR) Kathmandu: 18,000; Hotel limit (NPR) Other cities: 12,000" in text
    assert "$$a^{2}+b^{2}$$" in text


# -- PowerPoint (Phase 11) -------------------------------------------------------------


def make_pptx(path):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Q3 highlights"
    s.placeholders[1].text_frame.text = "Revenue reached NPR 42.5 million"
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Revenue by region"
    t = s.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(6), Inches(1)).table
    for (r, c), v in {(0, 0): "Region", (0, 1): "Growth", (1, 0): "Pokhara", (1, 1): "26%"}.items():
        t.cell(r, c).text = v
    s.notes_slide.notes_text_frame.text = "Pokhara grew fastest."
    prs.slides.add_slide(prs.slide_layouts[6])  # a blank slide
    prs.save(path)
    return path


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_powerpoint_is_one_text_per_slide_with_notes(tmp_path):
    s1, s2, s3 = layout.parse_file(make_pptx(tmp_path / "deck.pptx"))
    assert s1 == "# Q3 highlights\n\n- Revenue reached NPR 42.5 million"  # a body placeholder is a bullet list
    assert s2 == "# Revenue by region\n\n- Region: Pokhara; Growth: 26%\n\nSpeaker notes: Pokhara grew fastest."
    assert s3 == ""


# -- HTML (Phase 11) -------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html><html><head><title>Solar guide | Himal Energy</title><script>track()</script></head><body>
<header><nav><ul><li><a href="/">Home</a></li><li><a href="/shop">Shop</a></li></ul></nav></header>
<div class="cookie-banner">We use cookies. <button>Accept all</button></div>
<main><article><h1>Rooftop solar</h1><p>A 3 kW system costs about NPR 330,000. See <a href="/prices">prices</a>.</p>
<h2>Net metering</h2><table><tr><th>System</th><th>Price (NPR)</th></tr><tr><td>5 kW</td><td>520,000</td></tr></table>
</article></main><footer>© 2026 Himal Energy</footer></body></html>"""


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_html_keeps_the_article_without_menus_banners_or_link_targets(tmp_path):
    (tmp_path / "solar.html").write_text(HTML_PAGE)
    [text] = layout.parse_file(tmp_path / "solar.html")
    assert text == ("# Rooftop solar\n\nA 3 kW system costs about NPR 330,000. See prices.\n\n"
                    "## Net metering\n\n- System: 5 kW; Price (NPR): 520,000")


def test_clean_html_keeps_the_main_content():
    page = b"""<html><head><title>Guide</title></head><body><div id="vector-toc">Contents</div>
    <main id="content"><h1>Net metering</h1><div class="mw-portlet-lang">17 languages</div>
    <p>Credits roll over monthly.<sup class="reference">[1]</sup></p><span class="mw-editsection">edit</span>
    <div class="stock-price">NPR 8 per unit</div><div hidden>secret</div><div aria-hidden="true">icon</div>
    <ol class="references"><li>A source</li></ol></main><footer>(c) 2026</footer></body></html>"""
    out = layout.clean_html(page).decode()
    for kept in ("<title>Guide</title>", "Net metering", "Credits roll over monthly.", "NPR 8 per unit"):
        assert kept in out
    for gone in ("Contents", "17 languages", "[1]", "edit<", "secret", "icon", "A source", "(c) 2026"):
        assert gone not in out
