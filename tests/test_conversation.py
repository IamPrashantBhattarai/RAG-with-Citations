from citerag.conversation import REWRITE_SYSTEM, is_follow_up, rewrite_prompt, standalone_question
from citerag.generator import GenerationError

HISTORY = [("What monthly rent did the landlord agree to for the Lakeside office in Pokhara?",
            "The landlord agreed to a monthly rent of NPR 95,000 for the Lakeside office in Pokhara [1].")]


class FakeBackend:
    def __init__(self, reply="What is the deposit for the Lakeside office in Pokhara?", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.reply


def test_follow_ups_are_recognised_cheaply():
    for q in ("And how big is the deposit?", "What about Pokhara?", "How long does it run?", "Why?",
              "And the 5 kW one?", "What did they decide?"):
        assert is_follow_up(q), q
    for q in ("How much is budgeted for IT equipment for the Pokhara office?",  # "IT" is not "it"
              "Why are the dot products scaled by the square root of d_k?",
              "What monthly rent did the landlord agree to for the Lakeside office?"):
        assert not is_follow_up(q), q


def test_a_follow_up_is_rewritten_from_the_history():
    backend = FakeBackend()
    assert standalone_question("And how big is the deposit?", HISTORY, backend) == (
        "What is the deposit for the Lakeside office in Pokhara?")
    [(system, prompt)] = backend.calls
    assert system == REWRITE_SYSTEM
    assert "Assistant: The landlord agreed to a monthly rent of NPR 95,000" in prompt and "[1]" not in prompt
    assert prompt.endswith("Latest question: And how big is the deposit?")


def test_no_llm_call_without_history_or_for_a_standalone_question():
    backend = FakeBackend()
    assert standalone_question("And the deposit?", [], backend) == "And the deposit?"
    q = "How many days of paid annual leave do full-time employees get?"
    assert standalone_question(q, HISTORY, backend) == q
    assert backend.calls == []
    assert standalone_question(q, HISTORY, backend, always=True) != q and len(backend.calls) == 1


def test_bad_or_failed_rewrites_fall_back_to_the_question():
    q = "And the deposit?"
    for reply in ("", "Line one\nLine two", "x" * 500):
        assert standalone_question(q, HISTORY, FakeBackend(reply)) == q
    assert standalone_question(q, HISTORY, FakeBackend(error=GenerationError("quota"))) == q
    assert standalone_question(q, HISTORY, FakeBackend('Standalone question: "What is the deposit?"')) == (
        "What is the deposit?")


def test_only_recent_turns_and_short_answers_go_into_the_prompt():
    history = [(f"Question {i}?", "word " * 400) for i in range(5)]
    prompt = rewrite_prompt("And then?", history)
    assert "Question 0?" not in prompt and "Question 1?" not in prompt and "Question 4?" in prompt
    assert all(len(line) < 700 for line in prompt.splitlines())
