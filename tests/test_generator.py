from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from citerag.generator import NOT_FOUND, GenerationError, Generator, build_request
from citerag.store import Hit


def hit(i, text, source="syllabus.pdf", page=3, section="Evaluation", start=1000):
    return Hit(f"c{i}", text, source, page, page, section, start, start + len(text), 0.7)


HITS = [
    hit(0, "Full Marks: 100, Pass Marks: 45, Time: 3 Hrs. Group A has 20 questions."),
    hit(1, "Unit VII covers web services and server side scripting.", page=4, section=None),
]


def cite(doc, text_in_doc):
    start = HITS[doc].text.index(text_in_doc)
    return NS(type="char_location", document_index=doc, cited_text=text_in_doc,
              start_char_index=start, end_char_index=start + len(text_in_doc), document_title="t")


def text_block(text, citations=None):
    return NS(type="text", text=text, citations=citations)


class FakeStream:
    def __init__(self, message, events):
        self.message, self.events = message, events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, content=(), stop_reason="end_turn", error=None):
        self.requests = []
        self.error = error
        self.message = NS(content=list(content), stop_reason=stop_reason, model="claude-opus-5-5",
                          usage=NS(input_tokens=900, output_tokens=60))
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        events = []
        for b in self.message.content:
            if b.type == "text":
                events.append(NS(type="text", text=b.text))
                events += [NS(type="citation", citation=c) for c in b.citations or []]
        return FakeStream(self.message, events)


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def retrieve(self, question, k, sources):
        self.calls.append((question, k, sources))
        return self.hits


def make(content=(), hits=HITS, **client_kw):
    client = FakeClient(content, **client_kw)
    return Generator(FakeRetriever(hits), client, model="claude-opus-5-5", max_tokens=16000,
                     effort="medium"), client


ANSWER = [
    text_block("The pass mark is "),
    text_block("45 out of 100", [cite(0, "Pass Marks: 45")]),
    text_block(" and the exam lasts "),
    text_block("3 hours", [cite(0, "Time: 3 Hrs."), cite(0, "Pass Marks: 45")]),
    text_block("."),
]


# -- request ---------------------------------------------------------------------


def test_request_sends_one_cited_document_per_chunk_then_the_question():
    req = build_request("What are the pass marks?", HITS, "claude-opus-5-5", 16000, "medium")
    content = req["messages"][0]["content"]
    docs = content[:-1]
    assert len(docs) == 2 and all(d["type"] == "document" for d in docs)
    assert all(d["citations"] == {"enabled": True} for d in docs)
    assert docs[0]["source"] == {"type": "text", "media_type": "text/plain", "data": HITS[0].text}
    assert docs[0]["title"] == "syllabus.pdf, p. 3"
    assert "section: Evaluation" in docs[0]["context"] and "section" not in docs[1]["context"]
    assert content[-1] == {"type": "text", "text": "What are the pass marks?"}


def test_request_uses_effort_and_fallbacks_and_no_sampling_params():
    req = build_request("q", HITS, "claude-opus-5-5", 16000, "low")
    assert req["model"] == "claude-opus-5-5" and req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(req)


# -- answer ----------------------------------------------------------------------


def test_answer_numbers_citations_and_places_markers_after_claims():
    gen, _ = make(ANSWER)
    a = gen.answer("What are the pass marks?")
    assert a.text == "The pass mark is 45 out of 100[1] and the exam lasts 3 hours[2][1]."
    assert [c.number for c in a.citations] == [1, 2]  # repeated span reuses its number
    c = a.citations[0]
    assert (c.source, c.pages, c.section) == ("syllabus.pdf", "p. 3", "Evaluation")
    assert HITS[0].text[c.start_char:c.end_char] == c.cited_text == "Pass Marks: 45"
    assert c.doc_start_char == HITS[0].start_char + c.start_char
    assert a.found and a.usage == {"input_tokens": 900, "output_tokens": 60}


def test_segments_keep_uncited_text_visible_for_validation():
    a = make(ANSWER)[0].answer("q")
    assert [s.citations for s in a.segments] == [[], [1], [], [2, 1], []]


def test_streaming_callback_gets_text_and_live_markers():
    gen, _ = make(ANSWER)
    pieces = []
    gen.answer("q", on_text=pieces.append)
    assert "".join(pieces) == "The pass mark is 45 out of 100[1] and the exam lasts 3 hours[2][1]."


def test_no_relevant_chunks_means_claude_is_not_called():
    gen, client = make(ANSWER, hits=[])
    a = gen.answer("How do I bake bread?")
    assert client.requests == [] and a.text == NOT_FOUND and not a.found


def test_answer_without_citations_is_not_found():
    a = make([text_block("The documents don't say how long the course lasts.")])[0].answer("q")
    assert not a.found and a.citations == [] and a.text.startswith("The documents don't say")


def test_refusal_is_reported_not_parsed():
    a = make([text_block("partial")], stop_reason="refusal")[0].answer("q")
    assert a.stop_reason == "refusal" and not a.found and "declined" in a.text


def test_non_text_blocks_and_bad_document_indexes_are_ignored():
    bad = NS(type="char_location", document_index=7, cited_text="x", start_char_index=0,
             end_char_index=1, document_title="t")
    content = [NS(type="thinking", thinking=""), text_block("Claim", [bad])]
    a = make(content)[0].answer("q")
    assert a.text == "Claim" and a.citations == []


def test_retriever_gets_k_and_sources():
    gen, _ = make(ANSWER)
    gen.answer("q", k=3, sources=["syllabus.pdf"])
    assert gen.retriever.calls == [("q", 3, ["syllabus.pdf"])]


# -- errors ----------------------------------------------------------------------


def test_missing_api_key_gives_setup_hint():
    err = TypeError("Could not resolve authentication method. Expected one of api_key...")
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        make(ANSWER, error=err)[0].answer("q")


def test_connection_error_is_user_facing():
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(GenerationError, match="internet"):
        make(ANSWER, error=err)[0].answer("q")


def test_rate_limit_error_is_user_facing():
    resp = httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com"))
    err = anthropic.RateLimitError("rate limited", response=resp, body=None)
    with pytest.raises(GenerationError, match="Rate limited"):
        make(ANSWER, error=err)[0].answer("q")
