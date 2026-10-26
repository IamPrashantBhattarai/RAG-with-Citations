import pytest

from citerag.chunker import Chunker, _join_pages, _sentence_spans
from citerag.loaders import Page


def words(texts):
    return [len(t.split()) for t in texts]


def make(size=40, overlap=10, min_tokens=8):
    return Chunker(size, overlap, min_tokens, count_tokens=words)


def sentence(i, n=8):
    return f"Sentence {i} " + " ".join(f"w{i}x{k}" for k in range(n - 3)) + " end."


def long_paragraph(n=30):
    return " ".join(sentence(i) for i in range(n))


def test_chunks_respect_size_and_are_exact_source_slices():
    text = long_paragraph()
    chunks = make().chunk_document([Page(text, "doc.txt", 1)])
    assert len(chunks) > 1
    for c in chunks:
        assert c.n_tokens <= 40
        assert text[c.start_char : c.end_char] == c.text
        assert c.text.startswith("Sentence") and c.text.endswith("end.")


def test_every_sentence_is_covered():
    text = long_paragraph()
    joined = " ".join(c.text for c in make().chunk_document([Page(text, "d.txt", 1)]))
    for i in range(30):
        assert sentence(i) in joined


def test_mid_paragraph_cuts_overlap_by_whole_sentences():
    chunks = make().chunk_document([Page(long_paragraph(), "d.txt", 1)])
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char < a.end_char  # shared tail
        assert a.text.endswith(b.text[: a.end_char - b.start_char])


def test_paragraph_breaks_do_not_overlap():
    paras = [" ".join(sentence(p * 10 + i) for i in range(4)) for p in range(4)]
    chunks = make().chunk_document([Page("\n\n".join(paras), "d.txt", 1)])
    assert [c.text for c in chunks] == paras  # each 32-word paragraph is its own chunk


def test_headings_start_new_chunks_and_set_section():
    text = (
        "# Introduction\n\n" + " ".join(sentence(i) for i in range(3))
        + "\n\n## Methods\n\n" + " ".join(sentence(i) for i in range(10, 13))
    )
    chunks = make().chunk_document([Page(text, "paper.md", 1)])
    assert [c.section for c in chunks] == ["Introduction", "Methods"]
    assert chunks[1].text.startswith("## Methods")
    assert chunks[1].embed_text.startswith("paper > Methods\n\n")


def test_underlined_headings_are_detected():
    text = (
        "Overview\n========\n" + " ".join(sentence(i) for i in range(3))
        + "\n\nDetails\n-------\n\n" + " ".join(sentence(i) for i in range(10, 13))
    )
    chunks = make().chunk_document([Page(text, "notes.txt", 1)])
    assert [c.section for c in chunks] == ["Overview", "Details"]


def test_tiny_tail_is_merged_into_previous_chunk():
    text = long_paragraph(5) + " Short tail."
    chunks = make().chunk_document([Page(text, "d.txt", 1)])
    assert chunks[-1].text.endswith("Short tail.")
    assert all(c.n_tokens >= 8 for c in chunks)


def test_paragraph_spanning_a_page_break_keeps_both_pages():
    pages = [
        Page(long_paragraph(3) + " This sentence continues", "book.pdf", 4),
        Page("onto the next page. " + long_paragraph(1), "book.pdf", 5),
    ]
    text, offsets = _join_pages(pages)
    assert "continues onto" in text and offsets == [0, text.index("onto")]
    chunks = make(size=100).chunk_document(pages)
    assert (chunks[0].page_start, chunks[0].page_end) == (4, 5)
    assert chunks[0].id == "book-pdf-p4-c0"


def test_over_long_sentence_is_split_into_word_windows():
    text = " ".join(f"word{i}" for i in range(95)) + "."
    chunks = make().chunk_document([Page(text, "d.txt", 1)])
    assert all(c.n_tokens <= 40 for c in chunks)
    assert chunks[0].text.startswith("word0") and chunks[-1].text.endswith("word94.")


def test_sentence_splitter_handles_abbreviations_and_lists():
    text = "Use tools, e.g. Python. Dr. Smith agreed. J. Doe wrote it.\n- first item\n- second item"
    parts = [text[a:b] for a, b in _sentence_spans(text, 0, len(text))]
    assert parts == [
        "Use tools, e.g. Python.",
        "Dr. Smith agreed.",
        "J. Doe wrote it.",
        "- first item",
        "- second item",
    ]


def test_ids_are_unique_and_stable():
    pages = [Page(long_paragraph(), "My Notes.txt", 1), Page(long_paragraph(), "b.md", 1)]
    first = make().chunk_pages(pages)
    ids = [c.id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.id for c in make().chunk_pages(pages)]
    assert ids[0] == "my-notes-txt-p1-c0"


def test_empty_pages_and_invalid_settings():
    assert make().chunk_document([Page("   ", "d.txt", 1)]) == []
    with pytest.raises(ValueError):
        Chunker(chunk_size=600, count_tokens=words)  # can't fit a 512-token model
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, overlap=100, count_tokens=words)


def test_real_tokenizer_keeps_embed_text_within_model_limit():
    try:
        from citerag.chunker import hf_token_counter

        count = hf_token_counter()
    except Exception as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")
    text = "\n\n".join(long_paragraph(40) for _ in range(3))
    chunks = Chunker(count_tokens=count).chunk_document([Page(text, "real.txt", 1)])
    assert len(chunks) > 3
    for c, n in zip(chunks, count([c.embed_text for c in chunks])):
        assert c.n_tokens <= 384
        assert n + 2 <= 512
