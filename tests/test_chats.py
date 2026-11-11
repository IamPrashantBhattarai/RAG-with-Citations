import json

from citerag.chats import Chat, ChatStore, Turn, answer_from_dict, answer_to_dict
from tests.test_validator import validate


def test_a_checked_answer_survives_saving_and_loading(tmp_path):
    answer = validate("The pass marks are 45 [1]. Unit VII covers PHP [2].")
    again = answer_from_dict(json.loads(json.dumps(answer_to_dict(answer))))
    assert again == answer  # citations, segments, hits (with their text) and checks


def test_chats_are_saved_listed_reloaded_and_deleted(tmp_path):
    store = ChatStore(tmp_path)
    first = Chat()
    first.turns.append(Turn("What are the pass marks?", "What are the pass marks?", validate("The pass marks are 45 [1].")))
    store.save(first)
    second = Chat()
    second.turns.append(Turn("And the full marks?", "What are the full marks for Web Technology?",
                             validate("Full marks are 100 [1].")))
    store.save(second)
    assert [cid for cid, _ in store.list()] == [second.id, first.id]  # newest first
    loaded = store.load(first.id)
    assert loaded.title == "What are the pass marks?" and loaded.turns[0].answer.text.startswith("The pass marks")
    assert second.history() == [("What are the full marks for Web Technology?", "Full marks are 100 [1].")]
    store.delete(first.id)
    assert store.load(first.id) is None and [cid for cid, _ in store.list()] == [second.id]


def test_bad_ids_and_broken_files_are_ignored(tmp_path):
    store = ChatStore(tmp_path)
    for bad in ("../../etc/passwd", "ABCDEF123456", "", "123"):
        assert store.load(bad) is None
        store.delete(bad)  # never touches anything
    (tmp_path / "0123456789ab.json").write_text("{not json")
    assert store.load("0123456789ab") is None and store.list() == []


def test_long_titles_are_cut_at_a_word():
    chat = Chat()
    chat.turns.append(Turn("How does the Transformer give the model information about word order in a sentence?",
                           "x", validate("Positional encodings [1].")))
    assert chat.title == "How does the Transformer give the model information about…"
    assert Chat().title == "New chat"
