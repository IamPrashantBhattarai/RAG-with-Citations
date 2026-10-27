import pytest

from citerag.citations import InlineCitations
from citerag.generator import Answer, Generator, Segment
from citerag.retriever import tokenize
from citerag.store import Hit
from citerag.validator import (
    STRICT_EMPTY, Validator, clean, missing_terms, needs_citation, statement_spans, strict,
)


def hit(i, text, page=1):
    return Hit(f"c{i}", text, "syllabus.pdf", page, page, None, 0, len(text), 0.7)


HITS = [
    hit(0, "Full Marks: 100. Pass Marks: 45. Time: 3 Hrs. Eighty percent attendance is mandatory."),
    hit(1, "Unit VII: Web Services & Server Side Scripting (5 Hrs). 7.9. Introduction of Ruby.", page=3),
]


class FakeNLI:
    """Entailed when every content word of the statement is in the premise.

    `contradicts` lists statements whose premises contradict them.
    """

    def __init__(self, contradicts=()):
        self.contradicts, self.calls = set(contradicts), []

    def scores(self, pairs):
        self.calls.append(pairs)
        out = []
        for premise, hypothesis in pairs:
            entailed = set(tokenize(hypothesis)) <= set(tokenize(premise))
            out.append([0.95 if entailed else 0.05, 0.9 if hypothesis in self.contradicts else 0.0])
        return out and __import__("numpy").array(out)


def answer(text, hits=HITS):
    """A Gemini-style answer: [n] markers parsed into segments and quotes."""
    p = InlineCitations(hits)
    p.feed(text)
    p.close()
    return Answer("q", p.text, p.segments, p.citations, hits, bool(p.citations), invalid_markers=p.invalid)


def validate(text, nli=None, hits=HITS):
    a = answer(text, hits)
    a.validation = Validator(nli or FakeNLI(), threshold=0.5).validate(a)
    return a


def statuses(a):
    return [(c.text, c.status) for c in a.validation.checks]


def test_cited_and_supported_statement_is_ok():
    a = validate("The pass marks are 45 [1].")
    [c] = a.validation.checks
    assert (c.status, c.citations, c.issues, c.entailment) == ("ok", [1], [], 0.95)
    assert a.validation.ok and a.validation.summary() == "1/1 statements supported by their citations"


def test_uncited_claims_warn_and_say_whether_any_retrieved_passage_supports_them():
    a = validate("Unit VII has Web Services. The course is taught online. The pass marks are 45 [1].")
    checks = a.validation.checks
    assert [c.status for c in checks] == ["warn", "warn", "ok"]
    assert checks[0].issues == ["no citation, though syllabus.pdf, p. 3 supports it"]
    assert checks[1].issues == ["no citation, and no retrieved passage supports it"]
    assert a.validation.summary() == "1/3 statements supported by their citations, 2 warnings"


def test_lead_ins_headings_and_not_found_statements_need_no_citation():
    a = validate("Here are the marks:\n## Evaluation\n- The pass marks are 45 [1]\n"
                 "The documents don't mention the exam date.")
    assert statuses(a) == [
        ("Here are the marks:", "skip"),
        ("Evaluation", "skip"),
        ("The pass marks are 45", "ok"),
        ("The documents don't mention the exam date.", "skip"),
    ]


@pytest.mark.parametrize("text", [
    "The syllabus does not specify a textbook edition.",
    "The exam date is not mentioned.",
    "I couldn't find information about fees.",
    "There is no information about the lab schedule.",
    "The provided excerpts don't include the course fees.",
])
def test_not_found_statements(text):
    assert not needs_citation(text)


@pytest.mark.parametrize("text", [
    "Ruby is mentioned in Unit VII.",
    "The course does not allow re-exams.",
    "Students are not given a re-exam.",
])
def test_negative_claims_still_need_citations(text):
    assert needs_citation(text)


def test_wrong_number_is_flagged_even_if_the_words_match():
    a = validate("The pass marks are 40 [1].")
    [c] = a.validation.checks
    assert c.status == "warn" and c.issues[0] == 'not in the cited passage: "40"'


def test_numbers_match_number_words_ordinals_and_roman_numerals():
    assert missing_terms("80% attendance is required in Unit 7, the 1st unit of 5 Hrs", HITS) == []
    assert missing_terms("Unit 8 has 6 hours", HITS) == ["8", "6"]


def test_names_must_appear_in_the_cited_passage():
    assert missing_terms("It is offered by Tribhuvan University", [HITS[0]]) == ["Tribhuvan", "University"]
    assert missing_terms("It introduces Ruby and Web Services", [HITS[1]]) == []


def test_acronyms_match_initials_and_words_after_a_colon_are_not_names():
    css = hit(2, "Unit IV: Cascading Style Sheets (6 hr)")
    assert missing_terms("Unit IV covers CSS", [css]) == []
    assert missing_terms("Unit IV covers SCSS", [css]) == ["SCSS"]
    assert missing_terms("Ruby: Introduction of Ruby", [HITS[1]]) == []


def test_contradicted_statement_says_so():
    claim = "Pass marks are not required."
    a = validate("Pass marks are not required [1].", FakeNLI(contradicts={claim}))
    assert a.validation.checks[0].issues == ["the cited passage contradicts it"]


def test_invalid_markers_are_reported_in_the_summary():
    a = validate("The pass marks are 45 [1][9].")
    assert a.validation.invalid_markers == [9] and not a.validation.ok
    assert a.validation.summary().endswith("1 invalid citation marker dropped ([9])")


def test_all_statements_are_scored_in_one_nli_batch():
    nli = FakeNLI()
    validate("The pass marks are 45 [1]. Unit VII covers Ruby [2]. The course is online.", nli)
    assert len(nli.calls) == 1


def test_statement_spans_cover_sentences_and_list_lines_with_their_citations():
    segments = [Segment("Intro. "), Segment("The pass mark is 45", [1]), Segment(".\n- Ruby "),
                Segment("is taught", [2]), Segment(".")]
    plain = "".join(s.text for s in segments)
    assert [(plain[a:b], n) for a, b, n in statement_spans(segments)] == [
        ("Intro.", []), ("The pass mark is 45.", [1]), ("- Ruby is taught.", [2]),
    ]


def test_clean_strips_bullets_and_markdown():
    assert clean("*   **Web Services:** Introduction , *Examples* .") == "Web Services: Introduction, Examples."


def test_strict_mode_removes_warned_statements_and_renumbers_citations():
    a = strict(validate("Unit VII covers PHP [2]. The pass marks are 45 [1]."))
    assert a.text == "The pass marks are 45 [1]."
    assert [(c.number, c.cited_text) for c in a.citations] == [(1, "Pass Marks: 45.")]
    v = a.validation
    assert [c.text for c in v.removed] == ["Unit VII covers PHP."]
    assert v.summary() == "1/1 statements supported by their citations; strict mode removed 1 unsupported statement"
    plain = "".join(s.text for s in a.segments)
    assert [plain[c.start:c.end] for c in v.checks] == ["The pass marks are 45 ."]


def test_strict_mode_keeps_list_layout():
    a = strict(validate("Topics:\n- Web Services [2]\n- PHP and Node.js [2]\n- Ruby [2]"))
    assert a.text == "Topics:\n- Web Services [1]\n- Ruby [2]"  # each item quotes its own line


def test_strict_mode_with_nothing_verified_says_so():
    a = strict(validate("It is offered by Tribhuvan University [1]."))
    assert a.text == STRICT_EMPTY and a.citations == [] and not a.found
    assert a.validation.summary() == "No statement could be verified; strict mode removed 1 unsupported statement"


def test_strict_mode_leaves_a_clean_answer_unchanged():
    a = validate("The pass marks are 45 [1].")
    assert strict(a) is a


class FakeBackend:
    model = "fake"

    def __init__(self, text):
        self.text, self.on_text = text, "unset"

    def generate(self, question, hits, on_text=None):
        self.on_text = on_text
        if on_text:
            on_text(self.text)
        return answer(self.text, hits)


class FakeRetriever:
    def retrieve(self, question, k, sources):
        return HITS


def test_generator_validates_answers():
    g = Generator(FakeRetriever(), FakeBackend("The pass marks are 45 [1]."), Validator(FakeNLI()))
    assert g.answer("q").validation.ok


def test_strict_generator_streams_only_the_checked_answer():
    backend = FakeBackend("Unit VII covers PHP [2]. The pass marks are 45 [1].")
    g = Generator(FakeRetriever(), backend, Validator(FakeNLI()), strict=True)
    shown = []
    a = g.answer("q", on_text=shown.append)
    assert backend.on_text is None and shown == [a.text] == ["The pass marks are 45 [1]."]


def test_validation_can_be_turned_off():
    g = Generator(FakeRetriever(), FakeBackend("The pass marks are 45 [1]."), validate=False)
    assert g.answer("q").validation is None
