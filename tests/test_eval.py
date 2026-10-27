from types import SimpleNamespace

import pytest

from citerag import config
from eval.metrics import (
    Evidence,
    Question,
    answer_scores,
    covers,
    find_span,
    has_fact,
    load_questions,
    locate,
    problems,
    retrieval_scores,
)
from eval.retrieval import QUESTIONS, chunk_settings, gate_table

TEXT = "Exam rules.\nFull  Marks: 100,\nPass Marks: 45. There won’t be any re-exam."


def question(kind="single", quotes=("Pass Marks: 45",), facts=(("45",),), forbidden=()):
    evidence = [Evidence("syllabus.pdf", list(quotes))] if quotes else []
    q = Question("q1", "What are the pass marks?", kind, evidence, [list(f) for f in facts], list(forbidden))
    assert locate([q], {"syllabus.pdf": TEXT}) == []
    return q


def hit(start, end, source="syllabus.pdf"):
    return SimpleNamespace(source=source, start_char=start, end_char=end)


def rec(text, found=True, citations=(), hits=()):
    return {"text": text, "found": found, "citations": list(citations), "hits": list(hits)}


# -- locating quotes --------------------------------------------------------------


def test_find_span_ignores_case_whitespace_and_typographic_quotes():
    start, end = find_span(TEXT, "full marks: 100, pass MARKS: 45")
    assert TEXT[start:end] == "Full  Marks: 100,\nPass Marks: 45"
    start, end = find_span(TEXT, "there won't be")  # straight apostrophe vs. ’ in the text
    assert TEXT[start:end] == "There won’t be"


def test_find_span_missing_or_empty_quote():
    assert find_span(TEXT, "Pass Marks: 50") is None
    assert find_span(TEXT, "  ") is None


def test_locate_reports_quotes_not_in_the_document():
    q = Question("q9", "?", "single", [Evidence("syllabus.pdf", ["Pass Marks: 45", "no such text"])], [["45"]])
    assert locate([q], {"syllabus.pdf": TEXT}) == [
        "q9: quote not found in syllabus.pdf: 'no such text'"
    ]


def test_problems_in_the_question_set():
    docs = ["syllabus.pdf"]
    bad = [
        Question("a", "?", "single", [], []),
        Question("a", "?", "trivia", [], []),
        Question("b", "?", "off_topic", [Evidence("syllabus.pdf", ["x"])], []),
        Question("c", "?", "single", [Evidence("other.pdf", ["x"])], [["x"]]),
    ]
    assert problems(docs, bad) == [
        "a: answerable questions need evidence and facts",
        "a: duplicate id",
        "a: unknown kind 'trivia'",
        "b: off_topic questions can't have evidence",
        "c: other.pdf is not in the document list",
    ]


# -- retrieval --------------------------------------------------------------------


def test_a_chunk_covers_evidence_only_when_it_holds_a_whole_quote():
    q = question(quotes=["Pass Marks: 45", "Full  Marks: 100"])
    (s1, e1), (s2, e2) = q.evidence[0].spans
    e = q.evidence[0]
    assert covers("syllabus.pdf", s1, e1, e)
    assert covers("syllabus.pdf", s2 - 3, e2 + 3, e)  # the other place the fact appears
    assert not covers("syllabus.pdf", s1 + 1, len(TEXT), e)  # cut by a chunk boundary
    assert not covers("other.pdf", 0, len(TEXT), e)


def test_retrieval_scores():
    q = question()
    s, e = q.evidence[0].spans[0]
    hits = [hit(0, 10, "other.pdf"), hit(0, 11), hit(s - 5, e + 5)]
    assert retrieval_scores(q, hits, 1) == {"recall": 0.0, "all": False, "rr": 0.0, "top_doc": False}
    assert retrieval_scores(q, hits, 3) == {"recall": 1.0, "all": True, "rr": 1 / 3, "top_doc": False}


def test_retrieval_scores_with_several_pieces_of_evidence():
    q = Question("m", "?", "multi", [Evidence("syllabus.pdf", ["Exam rules"]),
                                     Evidence("syllabus.pdf", ["Pass Marks: 45"])], [["45"]])
    locate([q], {"syllabus.pdf": TEXT})
    scores = retrieval_scores(q, [hit(0, 11)], 5)
    assert scores["recall"] == 0.5 and not scores["all"] and scores["rr"] == 1.0


def test_gate_table_counts_each_kind():
    qs = [Question("a", "?", "single", [], []), Question("u", "?", "unanswerable", [], []),
          Question("o", "?", "off_topic", [], [])]
    rows, by_kind = gate_table({"a": 0.6, "u": 0.44, "o": 0.46}, qs)
    row = next(r for r in rows if r["threshold"] == 0.45)
    assert row == {"threshold": 0.45, "answerable_rejected": 0, "unanswerable_rejected": 1,
                   "off_topic_accepted": 1}
    assert by_kind == {"answerable": [0.6], "unanswerable": [0.44], "off_topic": [0.46]}


def test_chunk_settings_keep_the_default_proportions():
    assert chunk_settings(384) == (384, 64, 96)


# -- answers ----------------------------------------------------------------------


def test_has_fact_ignores_markers_and_markdown():
    assert has_fact("The pass marks are **45** [1].", ["45"])
    assert has_fact("Eighty  percent attendance", ["80%", "eighty percent"])
    assert not has_fact("The pass marks are 40 [1].", ["45"])


def test_answer_outcomes_for_answerable_questions():
    q = question(facts=[["45"], ["100"]])
    assert answer_scores(q, rec("Pass marks 45 of 100 [1]."))["outcome"] == "correct"
    assert answer_scores(q, rec("Pass marks are 45 [1]."))["outcome"] == "partial"
    assert answer_scores(q, rec("The documents don't mention pass marks."))["outcome"] == "refused"
    assert answer_scores(q, rec("I couldn't find anything.", found=False))["outcome"] == "refused"
    cited = [{"chunk_id": "c1", "source": "syllabus.pdf", "doc_start_char": 0, "doc_end_char": 5}]
    out = answer_scores(q, rec("Full marks are 100 [1]; the exact pass mark is not stated.", citations=cited))
    assert out["outcome"] == "partial"  # gave one fact
    out = answer_scores(q, rec("The excerpts do not give the pass marks [1].", citations=cited))
    assert out["outcome"] == "refused"
    assert answer_scores(q, rec("Pass marks are 40 [1]."))["outcome"] == "wrong"


def test_a_multi_document_answer_must_cite_every_document():
    q = Question("m", "?", "multi", [Evidence("a.pdf", ["x"]), Evidence("b.pdf", ["y"])], [["Adam"], ["Adafactor"]])
    one = {"chunk_id": "c", "source": "b.pdf", "doc_start_char": 0, "doc_end_char": 1}
    both = [one, {**one, "source": "a.pdf"}]
    assert answer_scores(q, rec("Adam [1] and Adafactor [2].", citations=[one]))["outcome"] == "partial"
    assert answer_scores(q, rec("Adam [1] and Adafactor [2].", citations=both))["outcome"] == "correct"


def test_unanswerable_questions_want_not_found_and_no_forbidden_facts():
    q = question(kind="unanswerable", quotes=(), facts=(), forbidden=["Tribhuvan"])
    assert answer_scores(q, rec("The documents don't say which university offers it."))["outcome"] == "correct"
    assert answer_scores(q, rec("Nothing found", found=False))["outcome"] == "correct"
    assert answer_scores(q, rec("It is offered by Tribhuvan University."))["outcome"] == "wrong"
    out = answer_scores(q, rec("The syllabus doesn't mention it, but it is Tribhuvan University."))
    assert out == {"forbidden": ["Tribhuvan"], "outcome": "wrong"}


def test_citation_scores():
    q = question()
    s, e = q.evidence[0].spans[0]
    hits = [{"chunk_id": "c1", "source": "syllabus.pdf", "start_char": 0, "end_char": len(TEXT)},
            {"chunk_id": "c2", "source": "other.pdf", "start_char": 0, "end_char": 50}]
    on_quote = {"chunk_id": "c1", "source": "syllabus.pdf", "doc_start_char": s, "doc_end_char": e}
    elsewhere = {"chunk_id": "c1", "source": "syllabus.pdf", "doc_start_char": 0, "doc_end_char": 11}
    other_doc = {"chunk_id": "c2", "source": "other.pdf", "doc_start_char": 0, "doc_end_char": 5}

    out = answer_scores(q, rec("45 [1]", citations=[on_quote], hits=hits))
    assert out["cites_evidence"] and out["quote_on_evidence"] and out["wrong_doc_citations"] == 0
    out = answer_scores(q, rec("45 [1]", citations=[elsewhere], hits=hits))
    assert out["cites_evidence"] and not out["quote_on_evidence"]  # right passage, wrong highlight
    out = answer_scores(q, rec("45 [1]", citations=[other_doc], hits=hits))
    assert not out["cites_evidence"] and out["wrong_doc_citations"] == 1


@pytest.mark.parametrize("text", [
    "The pass marks are 45 [1]. Attendance must be 80% [2].",
    "Intro. The marks are 45 [1][2] and quizzes twice [2].\n\n* **Exam:** three hours [1]\n* 80% [2]\nEnd.",
    "No citations at all here.",
    "The marks [1] [3] are shown. Then more [2, 3].",
])
def test_segments_rebuilt_from_text_match_the_streaming_parser(text):
    from citerag.citations import InlineCitations
    from citerag.store import Hit
    from eval.answers import segments_from_text

    passages = ["Full Marks: 100, Pass Marks: 45.", "Eighty percent attendance.", "Unit VII covers XML."]
    hits = [Hit(f"c{i}", t, "doc.pdf", 1, 1, None, 0, len(t), 0.5) for i, t in enumerate(passages)]
    parser = InlineCitations(hits)
    for i in range(0, len(text), 7):  # streamed in small pieces
        parser.feed(text[i:i + 7])
    parser.close()
    rebuilt = segments_from_text(parser.text)
    assert [(s.text, s.citations) for s in rebuilt] == [(s.text, s.citations) for s in parser.segments]


# -- the real question set --------------------------------------------------------


def test_question_set_is_well_formed():
    documents, questions = load_questions(QUESTIONS)
    assert problems(documents, questions) == []
    assert {q.kind for q in questions} == {"single", "multi", "unanswerable", "off_topic"}


def test_every_quote_is_found_in_the_documents():
    documents, questions = load_questions(QUESTIONS)
    if not all((config.DATA_DIR / d).is_file() for d in documents):
        pytest.skip("evaluation documents not in data/")
    from eval.retrieval import load_documents

    _, texts = load_documents(documents)
    assert locate(questions, texts) == []
