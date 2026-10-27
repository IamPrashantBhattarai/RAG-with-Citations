import re

import pytest

from citerag.citations import InlineCitations, locate_quote
from citerag.store import Hit


def hit(i, text, page=1):
    return Hit(f"c{i}", text, "syllabus.pdf", page, page, None, 500, 500 + len(text), 0.7)


HITS = [
    hit(0, "The course has eight units. Full Marks: 100. Pass Marks: 45. Time: 3 Hrs."),
    hit(1, "Unit VII covers web services. It also covers server side scripting with Ruby.", page=3),
]


def parse(text, pieces=None):
    p = InlineCitations(HITS)
    shown = "".join(p.feed(x) for x in (pieces or [text])) + p.close()
    return p, shown


def test_markers_become_exact_quotes_from_the_cited_excerpt():
    p, _ = parse("The pass mark is 45 out of 100 [1].")
    [c] = p.citations
    assert c.cited_text == "Full Marks: 100. Pass Marks: 45."  # claim uses both sentences
    assert HITS[0].text[c.start_char:c.end_char] == c.cited_text
    assert c.doc_start_char == 500 + c.start_char and c.support == 1.0


def test_citations_are_renumbered_by_first_appearance():
    p, shown = parse("Unit VII covers web services [2]. The pass mark is 45 [1].")
    assert shown == "Unit VII covers web services [1]. The pass mark is 45 [2]."
    assert [c.source for c in p.citations] == ["syllabus.pdf", "syllabus.pdf"]
    assert p.citations[0].page_start == 3


def test_live_text_matches_final_text_when_markers_split_across_chunks():
    pieces = ["The pass mark is 45 [", "1", "]. Unit VII covers server side scri", "pting [2]."]
    p, shown = parse(None, pieces)
    assert shown == p.text + "" == "The pass mark is 45 [1]. Unit VII covers server side scripting [2]."


def test_uncited_sentences_before_a_claim_stay_uncited_segments():
    p, _ = parse("Here is what I found. The pass mark is 45 [1].")
    assert [(s.text.strip(), s.citations) for s in p.segments] == [
        ("Here is what I found.", []),
        ("The pass mark is 45", [1]),
        (".", []),
    ]


def test_grouped_and_ranged_markers():
    for marker in ("[1][2]", "[1, 2]", "[1-2]"):
        p, shown = parse(f"Pass marks and server side scripting {marker}.")
        claim = [s for s in p.segments if s.citations][0]
        assert claim.citations == [1, 2], marker


def test_invalid_marker_numbers_are_dropped_and_recorded():
    p, shown = parse("The pass mark is 45 [1][9].")
    assert shown == "The pass mark is 45 [1]." and p.invalid == [9]


def test_a_marker_cites_only_its_own_line_not_the_heading_above():
    p, _ = parse("**2. Marks**\nThe pass mark is 45 [1]")
    assert [(s.text, s.citations) for s in p.segments] == [("**2. Marks**\n", []), ("The pass mark is 45 ", [1])]


@pytest.mark.parametrize("text", [
    "Topics:\n- Pass marks [1]\n- Server side scripting [2]",
    "**2. Marks**\n\nThe pass mark is 45 [1]. Unit VII covers scripting [2].",
    "Intro. The pass mark is 45 [1][9] and Ruby [2].\n",
])
def test_segments_keep_every_character_of_the_answer(text):
    p, _ = parse(text)
    assert "".join(s.text for s in p.segments) == re.sub(r"\[[\d,\s\-–]+\]", "", text)


def test_list_items_each_cite_their_own_line():
    p, _ = parse("- Pass Marks: 45 [1]\n- Server side scripting [2]")
    cited = [s for s in p.segments if s.citations]
    assert [s.citations for s in cited] == [[1], [2]]
    assert "Server side scripting" in cited[1].text and "Pass" not in cited[1].text


def test_unfinished_marker_at_end_is_plain_text():
    p, shown = parse("The pass mark is 45 [1")
    assert shown == "The pass mark is 45 [1" and p.citations == []


def test_no_markers_means_no_citations():
    p, shown = parse("The documents don't say how long the course lasts.")
    assert p.citations == [] and p.segments[0].citations == []


@pytest.mark.parametrize("claim, expected", [
    ("Pass marks are 45", "Pass Marks: 45."),
    ("It covers server side scripting with Ruby", "It also covers server side scripting with Ruby."),
])
def test_locate_quote_picks_best_sentence(claim, expected):
    text = HITS[0].text + " " + HITS[1].text
    s, e, support = locate_quote(claim, text)
    assert text[s:e] == expected and support > 0.5


def test_locate_quote_with_no_overlap_returns_whole_text_with_zero_support():
    s, e, support = locate_quote("photosynthesis in leaves", HITS[0].text)
    assert (s, e, support) == (0, len(HITS[0].text), 0.0)


TABLE = (
    "Full Marks: 100, Pass Marks: 45, Time: 3 Hrs\nNature of question\nTotal questions to be asked\n"
    "Group A: multiple choice\n20\n20\n20x1 = 20\n60%\nGroup B: Short answer type questions\n8\n6\n"
    "6x8 = 48\n60%\n100 100%\nEach student must secure at least 50% marks in internal evaluation in "
    "order to appear in the end semester\n\nexamination."
)


@pytest.mark.parametrize("claim, expected", [
    ("The pass marks are 45 out of full marks of 100",
     "Full Marks: 100, Pass Marks: 45, Time: 3 Hrs"),
    ("Students must secure at least 50% marks in the internal evaluation to appear in the end semester examination",
     "Each student must secure at least 50% marks in internal evaluation in order to appear in the end semester"),
])
def test_long_table_sentence_is_quoted_by_line_not_whole(claim, expected):
    s, e, support = locate_quote(claim, TABLE)
    assert TABLE[s:e] == expected and support >= 0.8


def test_next_items_section_number_is_trimmed_from_quote():
    text = "7.7. Web Services: Introduction, Characteristics, Examples 7.8. Server Side Scripting Languages 7.9."
    s, e, _ = locate_quote("web services characteristics", text)
    assert text[s:e] == "Web Services: Introduction, Characteristics, Examples"
