from types import SimpleNamespace as NS

import httpx
import pytest
from google.genai import errors

from citerag.gemini import GeminiBackend, build_prompt
from citerag.generator import DECLINED, GenerationError
from citerag.store import Hit

HITS = [
    Hit("c0", "Full Marks: 100. Pass Marks: 45. Time: 3 Hrs.", "syllabus.pdf", 3, 4, "Evaluation",
        0, 45, 0.7),
    Hit("c1", "Unit VII covers server side scripting.", "syllabus.pdf", 3, 3, None, 90, 128, 0.6),
]


def chunk(text=None, finish=None, usage=None, block=None):
    candidates = [NS(finish_reason=NS(name=finish))] if finish else []
    return NS(text=text, candidates=candidates, usage_metadata=usage,
              prompt_feedback=NS(block_reason=block) if block else None)


USAGE = NS(prompt_token_count=700, candidates_token_count=40, thoughts_token_count=25)


class FakeClient:
    """`errors` are raised by successive calls (None = succeed); the last one repeats."""

    def __init__(self, chunks=(), error=None, errors=None):
        self.calls, self.chunks = [], list(chunks)
        self.errors = errors if errors is not None else [error]
        self.models = NS(generate_content_stream=self._stream)

    def _stream(self, **kw):
        self.calls.append(kw)
        err = self.errors[min(len(self.calls) - 1, len(self.errors) - 1)]
        if err:
            raise err
        return iter(self.chunks)


def backend(chunks=(), error=None, errors=None):
    client = FakeClient(chunks, error, errors)
    b = GeminiBackend(client, model="gemini-3.8-flash", max_tokens=16000, thinking="low",
                      fallback_models=["gemini-3.6-flash", "gemini-3.5-flash-lite"])
    return b, client


def api_error(code, message="x"):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


STREAM = [chunk("The pass mark is 45 ["), chunk("1]. Unit VII covers server side scripting [2]."),
          chunk(finish="STOP", usage=USAGE)]


def test_prompt_numbers_excerpts_with_location_and_ends_with_question():
    prompt = build_prompt("What are the pass marks?", HITS)
    assert prompt.startswith("Source excerpts:")
    assert "[1] (syllabus.pdf, pp. 3-4, section: Evaluation)\nFull Marks: 100." in prompt
    assert "[2] (syllabus.pdf, p. 3)\nUnit VII" in prompt
    assert prompt.endswith("Question: What are the pass marks?")


def test_request_config_sets_system_prompt_thinking_and_token_cap():
    b, client = backend(STREAM)
    b.generate("q", HITS)
    cfg = client.calls[0]["config"]
    assert client.calls[0]["model"] == "gemini-3.8-flash"
    assert "[1] or [2][3]" in cfg.system_instruction
    assert cfg.max_output_tokens == 16000 and cfg.thinking_config.thinking_level.name == "LOW"
    assert cfg.temperature is None  # ignored by current Gemini models


def test_answer_has_streamed_text_exact_quotes_and_usage():
    b, _ = backend(STREAM)
    pieces = []
    a = b.generate("q", HITS, on_text=pieces.append)
    assert "".join(pieces) == a.text == "The pass mark is 45 [1]. Unit VII covers server side scripting [2]."
    assert [c.cited_text for c in a.citations] == ["Pass Marks: 45.", "Unit VII covers server side scripting."]
    assert a.found and a.stop_reason == "stop" and a.model == "gemini-3.8-flash"
    assert a.usage == {"input_tokens": 700, "output_tokens": 65}


def test_invalid_markers_are_kept_on_the_answer_for_validation():
    a = backend([chunk("The pass mark is 45 [1][7].", finish="STOP")])[0].generate("q", HITS)
    assert a.text == "The pass mark is 45 [1]." and a.invalid_markers == [7]


@pytest.mark.parametrize("chunks", [
    [chunk("partial", finish="SAFETY")],
    [chunk(block="PROHIBITED_CONTENT")],
])
def test_blocked_answers_are_declined(chunks):
    a = backend(chunks)[0].generate("q", HITS)
    assert a.text == DECLINED and not a.found


def test_missing_key_gives_setup_hint(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="aistudio.google.com"):
        GeminiBackend(None).generate("q", HITS)


@pytest.mark.parametrize("code, message, expected", [
    (429, "Resource has been exhausted", "Free-tier limit"),
    (400, "API key not valid. Please pass a valid API key.", "key was rejected"),
    (404, "models/x is not found", "not found"),
    (503, "overloaded", "overloaded right now"),
])
def test_api_errors_become_plain_messages(code, message, expected):
    err = api_error(code, message)
    with pytest.raises(GenerationError, match=expected):
        backend(error=err)[0].generate("q", HITS)


def test_network_error_is_user_facing():
    err = httpx.ConnectError("boom")
    with pytest.raises(GenerationError, match="internet"):
        backend(error=err)[0].generate("q", HITS)


def test_busy_model_falls_back_to_next_model():
    b, client = backend(STREAM, errors=[api_error(503), None])
    a = b.generate("q", HITS)
    assert [c["model"] for c in client.calls] == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert a.model == "gemini-3.6-flash" and a.found


def test_quota_exhausted_falls_back_too():
    b, client = backend(STREAM, errors=[api_error(429), api_error(429), None])
    assert b.generate("q", HITS).model == "gemini-3.5-flash-lite"


def test_bad_request_does_not_fall_back():
    b, client = backend(STREAM, errors=[api_error(400, "bad"), None])
    with pytest.raises(GenerationError, match="rejected the request"):
        b.generate("q", HITS)
    assert len(client.calls) == 1


def test_no_fallback_after_text_was_shown():
    def failing_stream():
        yield chunk("The pass mark is ")
        raise api_error(503)

    b, client = backend()
    client.models = NS(generate_content_stream=lambda **kw: (client.calls.append(kw), failing_stream())[1])
    shown = []
    with pytest.raises(GenerationError, match="stopped mid-answer"):
        b.generate("q", HITS, on_text=shown.append)
    assert len(client.calls) == 1 and shown == ["The pass mark is "]


def test_timeout_falls_back_to_next_model():
    b, client = backend(STREAM, errors=[httpx.ReadTimeout("slow"), None])
    assert b.generate("q", HITS).model == "gemini-3.6-flash"
