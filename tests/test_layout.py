import pytest

pytest.importorskip("docling_core")

from docling_core.types.doc import (BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size, TableCell,
                                    TableData)

from citerag import config, layout
from tests.test_loaders import body, make_pdf


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


def test_parse_pdf_is_cached_by_content_and_settings(tmp_path, monkeypatch, parse_cache):
    calls = []

    class FakeConverter:
        def convert(self, path):
            calls.append(path)
            return type("R", (), {"document": doc_with(("text", 1, "Hello."), ("text", 2, "World."))})

    monkeypatch.setattr(layout, "_converter", lambda formulas: FakeConverter())
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF one")
    assert layout.parse_pdf(pdf) == ["Hello.", "World."]
    assert layout.parse_pdf(pdf) == ["Hello.", "World."] and len(calls) == 1
    layout.parse_pdf(pdf, formulas="latex")  # different settings: parsed again
    pdf.write_bytes(b"%PDF two")  # changed file: parsed again
    layout.parse_pdf(pdf)
    assert len(calls) == 3 and len(list(parse_cache.iterdir())) == 3


def test_conversion_errors_become_layout_errors(tmp_path, monkeypatch):
    class Broken:
        def convert(self, path):
            raise RuntimeError("bad page tree")

    monkeypatch.setattr(layout, "_converter", lambda formulas: Broken())
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    with pytest.raises(layout.LayoutError, match="bad page tree"):
        layout.parse_pdf(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="CITERAG_PDF_FORMULAS"):
        layout.parse_pdf(tmp_path / "x.pdf", formulas="mathml")


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_real_docling_reads_a_pdf_page_by_page(tmp_path):
    pdf = make_pdf(tmp_path / "r.pdf", [body(1), [], body(3)])
    p1, p2, p3 = layout.parse_pdf(pdf)
    assert "apples" in p1 and p2 == "" and "comets" in p3
    assert "bridges" not in p1 + p3
