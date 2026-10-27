# Phase 6 — Citation Validation

**Status:** ✅ Done (2026-10-27): benchmarked, live-tested on the user's syllabus with Gemini, and fault-injection tested (10/10 injected errors caught) · Code: `citerag/validator.py` (+ small changes in `generator.py`, `gemini.py`, `citations.py`) · Tests: `tests/test_validator.py` (27), 136 passing in total

## Goal

A citation only shows that the model *pointed at* a passage, not that the passage says what the sentence claims. Phase 6 checks every statement of an answer against what it cites, marks each one ✅ or ⚠️ with a reason, and adds a summary line. An optional strict mode removes the statements that fail.

## Step 1 — What gets checked (the plan agreed on 2026-10-26)

| Check | How | Catches |
|---|---|---|
| **Uncited claims** | A statement with no `[n]` marker. Lead-ins ending in ":", headings, and "the documents don't say…" statements are skipped. The NLI model then reads every retrieved passage, so the warning can say "no citation, though syllabus.pdf p. 3 supports it" (missing citation) or "…and no retrieved passage supports it" (likely invented) | Facts added from general knowledge |
| **Numbers and names** | Every number and every proper name, acronym or mixed-case term in the statement must appear in a passage it cites. Numbers also match number words, ordinals and Roman numerals ("80" ~ "Eighty", "5" ~ "Fifth", "7" ~ "VII"); acronyms match initials ("CSS" ~ "Cascading Style Sheets") | "Pass marks are 40", "PHP and Laravel" |
| **Support (NLI)** | A local NLI (natural language inference) model reads each cited quote, the cited quotes joined, and each cited passage (the full chunk), and gives P(entailment) for the statement. The best of these must reach `NLI_THRESHOLD`. If a premise's P(contradiction) is ≥ 0.8, the warning says "contradicts" instead of "doesn't clearly support" | Wrong unit, swapped roles, negations, "6 hours" when 6 appears elsewhere in the passage |
| **Invalid markers** | `[n]` markers that pointed at excerpts that didn't exist (Gemini only; Phase 5 drops them from the text) are now kept on `Answer.invalid_markers` and reported in the summary | The model citing source [9] of 5 |

**Why check the full passage as well as the quote:** a quote is one sentence or table line. A correct statement often depends on context elsewhere in the chunk, e.g. the course name in the heading above the marks table. On the syllabus set, quote-only scoring had AUC 0.947 and quote-or-passage 0.989 (Step 2). With Gemini, the quote is chosen by our word-overlap code (Phase 5), so it can be off even when the passage is right.

**Why a lexical check as well as NLI:** NLI models are weak at exact numbers. In the fault-injection test (Step 5), the NLI model rated "pass marks are **40**" (0.66), "2 × **15** = **30** marks" (0.80) and "**PHP and Laravel**" (0.81) as supported; only the number/name check caught them. The two checks catch different errors.

**What counts as one statement:** the answer is split into lines first (LLM answers aren't hard-wrapped, so a line break ends a statement or list item), then into sentences with the chunker's own sentence splitter. A statement carries the citations of every cited segment it overlaps. Bold or `#` heading lines count as one unit and are skipped. A **cited** lead-in is still checked: Gemini writes "Unit VII covers the following topics [1]:", which is a claim.

## Step 2 — Choosing the NLI model (benchmark)

**Test data**
1. **RAGTruth QA** (`wandb/RAGTruth-processed`, test split, ungated): real answers from 6 LLMs to questions over retrieved web passages, with every hallucinated span annotated by people. Each answer sentence becomes one claim, labelled hallucinated if it overlaps an annotated span. Contexts ≤ 300 words only (about the size of our 384-token chunks). The answers' own "(Passage 2)" references were stripped, because our citations are `[n]` markers that never reach the check. 795 sentences, 265 hallucinated.
2. **Syllabus set**: 81 claims hand-written against the 7 syllabus chunks, 47 true and 34 false. The false ones are the mistakes that matter: wrong numbers, wrong unit, negations, and facts from general knowledge. The script is in the session scratchpad and isn't committed (same as the earlier benchmarks).
3. **12 real Gemini answers** on the syllabus (41 checked statements; all answers were correct, so every warning is a false alarm).

Ruled out up front: **LLM-AggreFact** (the standard grounding benchmark) is gated, needing a Hugging Face login and its licence accepted. **Vectara HHEM** needs `trust_remote_code`, which runs code from the model repo. **MiniCheck** ships only pickle `.bin` weights (1.7 GB).

**NLI model alone** (AUC = how well the score separates true from false claims, 1.0 = perfectly)

| Model | Download | RAGTruth AUC | Syllabus AUC (quote / passage / best of both) | ms per pair (MPS) |
|---|---|---|---|---|
| cross-encoder/nli-deberta-v3-base | 738 MB | 0.717 | 0.926 / 0.812 / 0.941 | 96 |
| **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli** | 369 MB | 0.728 | 0.947 / 0.987 / 0.989 | 71 |
| tasksource/deberta-small-long-nli | 568 MB | **0.808** | 0.912 / 0.941 / 0.923 | **59** |
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 870 MB | 0.790 | 0.946 / 0.989 / **0.996** | 227 |

**Full decision (number/name check + NLI ≥ 0.5)** (balanced accuracy = average of "true claims passed" and "false claims caught")

| Model | RAGTruth bal. acc. (caught / false alarms) | Syllabus bal. acc. (caught / false alarms) | Real answers falsely warned | Time per answer |
|---|---|---|---|---|
| tasksource-small | **0.752** (206/265, 145/530) | 0.907 (32/34, 6/47) | 6/41 | **0.43 s** |
| **MoritzLaurer base** | 0.708 (185/265, 150/530) | 0.953 (33/34, 3/47) | 3/41 | 0.60 s |
| MoritzLaurer large | 0.727 (183/265, 125/530) | **0.960** (32/34, 1/47) | **2/41** | 1.69 s |

**Decision: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, threshold 0.5.**
- On the user's kind of documents (the syllabus set and the real answers), it's close to the large model and clearly better than tasksource.
- tasksource wins on RAGTruth's web text, but it fails on **tables**. It flagged all four exam-format rows ("Group A: 20 questions … 20 × 1 = 20 marks") and "the exam lasts 3 hours" (from `Time: 3 Hrs`). Syllabi are full of tables.
- The large model is a little more accurate everywhere but about 3× slower. It's one setting away: `CITERAG_NLI_MODEL=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
- Threshold 0.5: best on the syllabus set (0.3: 0.939, 0.5: 0.953, 0.7: 0.947) and within 0.005 of the best on RAGTruth. The MoritzLaurer models are well calibrated (best threshold ≈ 0.5–0.6); cross-encoder/nli-deberta-v3-base wasn't (best threshold ≈ 0.003).
- Contradiction cut-off 0.8: on the syllabus set, most false claims scored ≥ 0.83, while the true claims that were wrongly flagged scored 0.06, 0.61 and 0.93. Below 0.8 the message is the softer "doesn't clearly support it".
- No new package: `sentence_transformers.CrossEncoder` (already installed) loads it, in the same way the optional reranker loads.

Before tuning, the number/name check flagged 13% of correct RAGTruth sentences. Most were "(Passage 2)" references, and the rest were capitalised words after a colon ("…culture: Take some time"). After stripping those references and skipping the word after ":"/";" it flags 3.4% (18/530). On the syllabus set, "CSS" was flagged against "Cascading Style Sheets" until acronyms could match initials (now 0/47 false alarms).

## Step 3 — Strict mode

`CITERAG_STRICT=1` or `--strict` removes every ⚠️ statement, renumbers the citations that are left ([1], [2]… with no gaps), and drops citations only the removed statements used. If nothing verifiable is left, the answer becomes "None of the answer's statements could be verified against your documents." (`found=False`).
- Nothing streams in strict mode. Text already shown can't be taken back, so the checked answer is passed to `on_text` once, at the end.
- The answer text is cut, not re-rendered: Gemini places markers as `45 [1].` and Claude as `45[1].`, and re-rendering turned one into the other (`45[1] .`), which a test caught. `_cut_text` aligns the displayed text with the plain text and deletes the removed spans, and each marker goes with the character before it.
- `Validation.removed` keeps what was removed and why, so the summary says e.g. "strict mode removed 2 unsupported statements".

## Step 4 — Wiring and output

- `Answer.validation` (`Validation`: `checks`, `claims`, `warnings`, `ok`, `summary()`, `invalid_markers`, `removed`). Each `Check` has `text`, `citations`, `status` (`ok`/`warn`/`skip`), `issues`, `entailment`, and `start`/`end` in the plain answer text (for the UI).
- `Generator(validate=config.VALIDATE, strict=config.STRICT)` validates every LLM answer. The NLI model loads on first use. All of an answer's (premise, statement) pairs are scored in **one batch**. Answers with nothing retrieved make no LLM call and aren't validated.
- Config: `CITERAG_VALIDATE` (default on), `CITERAG_STRICT` (default off), `CITERAG_NLI_MODEL`, `CITERAG_NLI_THRESHOLD` (0.5).

```bash
python -m citerag.generator "What are the pass marks?"            # answer, sources, then checks
python -m citerag.generator "..." --strict                        # only verified statements
python -m citerag.generator "..." --no-validate                   # skip the checks
```

```
Checks:
  ⚠️  The internal evaluation is worth a total of 20 marks and is calculated using the following weightage:[1]
       the cited passage doesn't clearly support it
  ✅ Assignments: 20%[2]
  ...
  5/6 statements supported by their citations, 1 warning
```

The CLI loads the NLI model before the answer streams. Loading it afterwards printed a torch log line in the middle of the answer's last line.

## Step 5 — Live test and fault injection (2026-10-27, Gemini free tier)

**Fixes from the first real answers**
1. A bold numbered heading (`**1. Requirement to Appear…**`) was split at "1. " into two pieces, and the second piece was flagged as an uncited claim. → Heading lines are now detected before the sentence split and skipped.
2. **Phase 5 marker parsing:** a `[n]` marker cited everything back to the previous sentence end, including a heading line above it (`**2. Weightage**\nThe internal evaluation…[3]`), so the heading was checked as a claim. → `_split_last_sentence` now looks only at the marker's own line. A first version dropped the line break before cited list items, which a test caught. A new test checks that the segments always contain every character of the answer.
3. Cited lead-ins ("…covers the following topics [1]:") were skipped as lead-ins. → Anything the model cites is checked.

**Results:** 3 of 41 statements in the 12 real answers were wrongly warned, all borderline:
- the Unit VII lead-in (entailment 0.447)
- "falling below 80% results in NQ status" (the PDF text reads "to **a ttend**"; with the space removed the score goes from 0.17 to 0.73)
- "the course specifically focuses on Ruby and Rails" (an inference).

A tricky question ("which languages will I learn, and which code editor should I use?") got a correct answer that said the editor wasn't in the documents. That statement was skipped, and 4/4 and 5/5 statements passed with and without `--strict`.

**Fault injection:** 10 realistic errors were edited into the real answers before validation: wrong numbers (pass marks 40, attendance 75%, XML unit 6 hours, 2 × 15 = 30), a wrong edition ("Fifth"), invented frameworks (PHP and Laravel), a made-up uncited fact ("offered by Tribhuvan University"), swapped examiner roles, "two hours", and an invented re-exam rule. **All 10 were flagged.** Strict mode removed exactly the 2 tampered lines from the practical-exam answer and kept the list layout.

## Tests (`tests/test_validator.py`, 27)

A fake NLI model (entailed when the statement's words are all in the premise) keeps the tests offline and instant. They cover:
- supported and uncited statements, including "supported by an uncited passage"
- lead-ins, headings and not-found statements skipped; negative claims still checked
- wrong numbers; number words, ordinals and Roman numerals; names; acronyms; words after a colon
- contradiction wording; invalid markers in the summary; one NLI batch per answer
- statement splitting; markdown cleanup
- strict mode: removal, renumbering, list layout, nothing verified, a clean answer unchanged
- the Generator: validates by default, strict mode streams only the checked answer, validation can be turned off

Also new: `test_citations.py` (a marker cites only its own line; segments keep every character) and `test_gemini.py` (invalid markers reach the `Answer`).

## Known limitations

- **PDF extraction artifacts** ("a ttend", "e valuated", "HT ML5") lower NLI scores and can cause false warnings. Fixing them belongs in the Phase 1 loader (re-joining split words).
- **Tables are the weakest case.** Even the chosen model warned on "worth a total of 20 marks", taken from the table row `Total Internal 100% 20`.
- **A wrong quote can still pass.** When the full passage supports a statement but the quote doesn't, the statement passes. That's right for the citation but leaves an imprecise highlight. Example: "Internal Exams: 50%" got a quote from the wrong part of an interleaved table.
- Number words in the *statement* aren't checked lexically ("two hours" relies on NLI), because words like "one" are often not numbers.
- About 0.6 s per answer on Apple MPS for a typical 4–8 statement answer, plus about 4 s to load the model once.
- RAGTruth balanced accuracy is about 0.71, so on out-of-domain web text the checks are a useful signal, not proof.
