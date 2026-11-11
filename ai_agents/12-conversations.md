# Phase 12 — Multi-turn chat

**Status:** ✅ Done (2026-11-11) · Code: `citerag/conversation.py` (new), `citerag/chats.py` (new), `citerag/generator.py` (`complete`), `citerag/gemini.py` (`complete`), `citerag/config.py`, `app.py`; `eval/followups.json` and `eval/followups.py` (new) · Tests: `tests/test_conversation.py` (5), `tests/test_chats.py` (4), +1 `tests/test_generator.py`, +2 `tests/test_gemini.py`, +2 `tests/test_ui.py`; 255 passing in total

**Outcome in one line:** follow-up questions ("And how big is the deposit?") are rewritten into standalone questions before searching, and chats are saved and reopen after a reload. On 12 test follow-ups, answers went from 9/12 correct (searched as typed) to 12/12.

## Goal

The user picked this from the "industry grade" list: Phase 7 noted that each question was answered on its own and the chat vanished on reload.

## Step 1 — Design

**Rewrite, then answer as usual.** A follow-up can't be searched as typed; what "its" or "what about" refers to is in the earlier turns. The LLM rewrites it, from the last three turns (questions plus the first 600 characters of each answer, without `[n]` markers), into a standalone question. That question then goes through the unchanged pipeline: search, a cited answer, the checks. **The chat history never reaches the answering prompt**, so every statement is still checked against the cited passages and nothing can be "remembered" into an answer. The app shows the rewrite under the question ("Searched for: …"), so a wrong rewrite is visible.

**Only when needed.** A question that stands on its own skips the extra LLM call (free-tier quotas are small): `is_follow_up` is a cheap check for words that point back ("it", "they", "that", "what about…", "and…") or a question of four words or fewer. On the 126 standalone eval questions it flagged 7 at first. Two were fixed: "Why …" questions stand alone, and "IT equipment" isn't the pronoun "it". It now flags 3 ("this course", "does it contain", "Is this the first return"). Those cost one call each, and the rewrite is told to return them unchanged.

**Guards on the rewrite** (`standalone_question`):
- the prompt treats the conversation as data, since earlier answers quote the documents
- the reply must be a single line under 400 characters, with a "Standalone question:" prefix or quotes stripped
- if it fails, is empty, or is longer, the question is searched as typed
- an LLM error (quota) also falls back to the question as typed

**`Backend.complete(system, prompt)`** is a short plain-text call on both backends:
- **Claude:** `client.beta.messages.create` with `output_config={"effort": "low"}`. Thinking can't be turned off on Claude Opus 5.5 and effort defaults to `medium`, so low is set explicitly. `max_tokens` is 4,096, room for the hidden thinking. It uses the same server-side refusal fallback as answers, and a refusal or a cut-off reply returns "".
- **Gemini:** `generate_content` at thinking level `LOW`, 1,024 tokens, and the same next-model fallback as answers.
- The Anthropic error mapping now lives in one place (`_anthropic_errors`), shared by streaming answers and `complete`.

**Saved chats** (`citerag/chats.py`): one JSON file per chat in `.chats/` (git-ignored; `CITERAG_CHATS_DIR`). Each turn stores the question as typed, what was searched, and the **whole answer**: citations, segments, checks and the passages given to the model. An old chat therefore still shows its sources after a re-index. Writes go through a temp file and a rename. Chat ids are 12 hex characters, and anything else is refused before a file is touched, because the id comes from the URL.

**App:**
- the chat id is in the address (`?chat=…`), so a reload or a bookmark reopens it
- the sidebar's **Chats** section has *New chat*, the recent chats by their first question, and *Delete this chat*; it replaces *Clear chat*
- each turn shows "Searched for: …" when the question was rewritten

## Step 2 — Evaluation

`eval/followups.json`: 12 two-turn chats over the 8 evaluation documents. Each first turn is a question from `questions.json` with **the answer CiteRAG actually gave it** in the saved runs, so the history is realistic. The follow-ups cover:
- pronouns: "How long does it run?", "How many parameters does it have?"
- "what about": "What about Pokhara?", "What about a Director?", "What about English-to-French?"
- "and …": "And sick leave?", "And the 5 kW one?", "And at the beginning of the year?"
- an implicit reference: "What is the target for the next quarter?"
- **a topic change** that the history must not leak into (fu-12)

`python -m eval.followups` searches each follow-up three ways, **raw** (as typed), **gated** (rewritten when it looks like a follow-up, which is what the app does) and **always** (rewritten whenever there's history), then answers it.

Results (`gemini-3.5-flash-lite`, reranker on, top 5):

| | Raw (as typed) | **Gated (the app)** | Always rewrite |
|---|---|---|---|
| Evidence in the top 5 | 11/12 | **12/12** | 12/12 |
| Answers correct | 9/12 | **12/12** | — |
| LLM calls for rewriting | 0 | 11 | 12 |

- **The rewrites were right in all 11 cases.** For example, "What about Pokhara?" became "What was the Pokhara region's revenue in Q3?", "And the 5 kW one?" became "How much does a complete 5 kW rooftop solar system cost installed?", and "How many parameters does it have?" became "How many parameters does ViT-G/14 have?". The topic change (fu-12) came through unchanged, with no history leaking in.
- **Raw follow-ups often still find the evidence** (11/12) because, with eight documents, a word like "deposit" or "sick leave" is nearly unique. **But the answers show the difference:**
  - fu-02 "How long does it run?" was answered about **the Transformer's training time**: a confident, cited answer about the wrong document.
  - fu-06 "What about a Director?" found nothing and was declined.
  - fu-10 "How many parameters does it have?" got "unclear which model 'it' refers to".

  The first failure is the dangerous kind: it looks well sourced.
- **Gated vs always:** the one follow-up the check misses (fu-04, "What is the target for the next quarter?", with no pointing word) was answered correctly anyway: "target" was enough for search. Always rewriting found nothing more here and costs a call on every question in a chat, so the app keeps the check.
- **Checker:** 1 warning in 12 gated answers (fu-10 cites two passages for "nearly two billion" and "1,843 million"; both are right).
- Rewrites and answers are saved in `eval/runs/followups-lite.json` (git-ignored); re-running reuses them.

## Step 3 — Tests

- `test_conversation.py` (5):
  - follow-ups recognised, and standalone questions not flagged ("IT equipment", "Why are …")
  - the rewrite prompt built from history, without `[n]` markers
  - no LLM call without history or for a standalone question, unless forced
  - bad replies (empty, several lines, too long, an LLM error) fall back to the question as typed, and a "Standalone question:" prefix and quotes are removed
  - only the last three turns go into the prompt, each answer cut short
- `test_chats.py` (4):
  - a checked answer survives saving and loading exactly
  - chats are saved, listed newest first, reloaded and deleted
  - bad ids (`../../etc/passwd`) and broken files are ignored
  - titles are cut at a word
- `test_generator.py` (+1): Claude's `complete()` sends effort `low`, the refusal fallback and no `thinking` field, reads only the text blocks, and gives "" on a refusal or cut-off.
- `test_gemini.py` (+2): `complete()` falls back to the next model when busy; a blocked or cut-off reply gives ""; other errors are reported.
- `test_ui.py` (+2, headless app):
  - a follow-up is shown as "Searched for: …", only the follow-up was rewritten, and the chat is saved with both turns
  - a saved chat reopens from the address in a fresh session, listed under Chats

## Limitations

- **One rewrite per follow-up costs one more LLM call.** On the Gemini free tier, a chat of many follow-ups uses quota faster.
- **The check for follow-ups is a heuristic.** An implicit follow-up with no pointing word ("What is the target for the next quarter?") is searched as typed.
- **Chats are local and single-user:** there are no accounts, so anyone using this computer's browser sees them. That's the next "industry grade" item (login and permissions).
- **The answer doesn't see the conversation,** by design. "Summarise what we discussed" isn't supported.
