# Phase 8 — Evaluation and tuning

**Status:** ✅ Done (2026-10-27) · Code: `eval/` (`questions.json`, `metrics.py`, `retrieval.py`, `answers.py`); changes in `citerag/config.py`, `citerag/validator.py`, `app.py` · Tests: `tests/test_eval.py` (20), 18 new in `tests/test_validator.py`; 191 passing in total

**Outcome in one line:** the reranker is now on by default (answers 50/55 → 53/55 correct), and the citation checker's false alarms dropped from 28% to 8% of true statements; chunk size, k and the relevance gate were confirmed.

## Goal

Replace "it seems to work" with numbers measured on the user's own documents, and use them to check every setting chosen in earlier phases from public benchmarks or reasoning: chunk size, search method, reranker, how many passages the LLM sees, and the "not in the documents" gate.

## Step 1 — The documents

The syllabus alone is 7 chunks, and the LLM is given 5, so retrieval would almost always look perfect. The user added two papers to `data/`:

| Document | Pages | Chunks (384 tokens) | What makes it hard |
|---|---|---|---|
| `bsccsit-syllabus-webtechnology.pdf` | 4 | 7 | Tables extracted as runs of text, split words ("a ttend") |
| `NIPS-2017-attention-is-all-you-need-Paper.pdf` | 11 | 30 | Equations and result tables; symbols run into words ("dimensiondmodel") |
| `Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.pdf` | 10 | 46 | Two columns; figure axis numbers mixed into the text; facts repeated in abstract, results and tables |

Both papers are about Transformers, Adam, attention heads and BLEU/accuracy numbers, so they compete with each other for the same questions. That is a realistic test of choosing the right document, not just the right passage.

## Step 2 — The question set (`eval/questions.json`)

78 questions, written by reading the text exactly as the loader extracts it:

| Kind | n | Right outcome |
|---|---|---|
| `single` | 52 | correct facts, cited (16 syllabus, 18 per paper) |
| `multi` | 3 | facts from both papers |
| `unanswerable` | 8 | "the documents don't say", without tempting outside facts (e.g. "Which university offers this course?" must not say Tribhuvan) |
| `off_topic` | 15 | stopped by the gate, or "not in the documents" |

Each answerable question lists:
- **evidence**: exact quotes from the document. **Why quotes, not chunk IDs:** chunk IDs change with the chunk size, quotes don't, so one question set works for every configuration. A retrieved chunk is a hit when it contains a whole quote (`eval/metrics.py: covers`).
- **facts**: what a correct answer must contain, as groups of alternatives (`["80%", "80 percent", "eighty percent"]`). Every group must match. Deterministic, free, and repeatable, unlike an LLM judge.

**Honest caveat:** questions written by reading the passage tend to reuse its words, which favours keyword search. Some were paraphrased on purpose ("How does the Transformer give the model information about word order?" for positional encoding), but real users will paraphrase more.

## Step 3 — First run: the gold labels were wrong, not the system

The first retrieval run counted "What top-1 accuracy does ViT-G/14 reach on ImageNet?" as a miss for all four search methods. The number 90.45% appears **four times** in the paper (abstract, introduction, Table 1, Section 2.5), and the label listed only the abstract. A system that found Section 2.5 was right and was scored wrong.

**Fix:** a piece of evidence can list every place the fact appears (`"quote": [...]`), and any of them counts. A script listed every repeated fact in the documents. 11 questions got alternatives: for example, "two billion parameters" appears 5 times and 84.86% 4 times. A **reachability check** was added too: if no chunk contains a quote whole (a chunk boundary cut it), no search method can find it, and that is reported as a chunking problem rather than a retrieval miss.

**Lesson:** check the gold labels before trusting a benchmark.

## Step 4 — Retrieval benchmark (`python -m eval.retrieval`)

Local only, no API calls. Every configuration gets its own index under `eval/.cache/`; the app's index in `.chroma/` is never touched. Recall@k is the share of evidence found in the top k chunks; **all@5** (every piece of evidence is among the 5 chunks the LLM sees) is the number that matters most. 55 answerable questions.

### Chunk size × search method

Overlap and minimum size scale with the chunk size (384/64/96 is the current default).

| Chunk size | Chunks | Method | R@1 | R@5 | all@5 | MRR | Words given to the LLM (k=5) | Time / question |
|---|---|---|---|---|---|---|---|---|
| 128 | 237 | dense | 0.57 | 0.81 | 0.78 | 0.69 | 354 | 8 ms |
| | | keyword | 0.65 | 0.86 | 0.84 | 0.78 | 354 | <1 ms |
| | | hybrid | 0.61 | 0.85 | 0.82 | 0.75 | 358 | 9 ms |
| | | rerank (30) | 0.77 | 0.91 | 0.89 | 0.86 | 337 | 0.5 s |
| 256 | 125 | dense | 0.52 | 0.85 | 0.84 | 0.65 | 641 | 8 ms |
| | | keyword | 0.72 | 0.88 | 0.85 | 0.82 | 649 | <1 ms |
| | | hybrid | 0.60 | 0.92 | 0.91 | 0.74 | 661 | 9 ms |
| | | rerank (30) | 0.80 | 0.93 | 0.91 | 0.88 | 602 | 0.9 s |
| **384** | 83 | dense | 0.45 | 0.84 | 0.80 | 0.61 | 985 | 8 ms |
| | | keyword | 0.68 | 0.93 | 0.91 | 0.81 | 957 | <1 ms |
| | | hybrid (default) | 0.56 | 0.91 | 0.89 | 0.73 | 985 | 9 ms |
| | | rerank (30) | 0.84 | 0.96 | 0.95 | 0.91 | 942 | 1.4 s |
| 448 | 73 | dense | 0.40 | 0.80 | 0.76 | 0.58 | 1061 | 8 ms |
| | | keyword | 0.72 | 0.92 | 0.89 | 0.82 | 1064 | <1 ms |
| | | hybrid | 0.59 | 0.91 | 0.89 | 0.74 | 1093 | 9 ms |
| | | rerank (30) | 0.77 | 0.95 | 0.93 | 0.87 | 1022 | 1.5 s |

**Findings**
- **384 tokens stays.** No size beats it by more than one question out of 55, which is noise. 128 is worst: one syllabus quote (the textbook) is cut by a chunk boundary at every overlap, and tables lose their header rows.
- **Dense search alone is the weakest method here**, unlike on SciFact/NFCorpus in Phase 4. Questions name exact things (ViT-G/14, BLEU, P100, Adafactor) that keyword search matches directly. Part of this is the caveat in Step 2.
- **Hybrid is pulled between its two halves.** RRF gives dense and keyword equal weight, so a weak dense ranking drags keyword's top results down (R@1 0.56 vs 0.68 for keyword alone).
- **The reranker is clearly best**, the opposite of the Phase 4 public benchmarks, which is why Phase 4 kept it as a switch to re-test on real questions.

### How many candidates to rerank

At 384 tokens:

| Candidates reranked | all@5 | R@1 | MRR | Time / question |
|---|---|---|---|---|
| **10 (new default)** | **0.96** | 0.84 | 0.91 | **0.45 s** |
| 15 | 0.96 | 0.84 | 0.91 | 0.68 s |
| 20 | 0.95 | 0.84 | 0.91 | 0.93 s |
| 30 (Phase 4 default) | 0.95 | 0.84 | 0.91 | 1.40 s |

Reranking the top 10 is as accurate as reranking 30 and 3× faster. With a longer list, the reranker can promote a wrong chunk from far down: "How many layers does the base Transformer encoder have, and how deep is ViT-L/16?" lost its evidence at 20 and 30 candidates.

**Is the reranker's gain real or noise?** A paired sign test over the 55 questions, reranker (10 candidates) vs. hybrid (the default):

| Measure | Reranker better | Hybrid better | p (two-sided) |
|---|---|---|---|
| right chunk ranked 1st | 15 | 1 | 0.0005 |
| reciprocal rank | 20 | 2 | 0.0001 |
| all evidence in the top 5 | 4 | 0 | 0.125 |

Ranking is significantly better. In the top 5 the reranker never loses a question and wins four, but four questions alone isn't conclusive. Step 5 checks whether better ranking gives better answers.

### What still fails

- **"How many parameters does the big Transformer model have?"** (every method). The answer is the last number of a table row, `big 6 1024 4096 16 0.3 300K 4.33 26.4 213`, and the "params ×10⁶" header is far away in the same chunk. Nothing in the row says "parameters".
- **"Which optimizer did the Transformer paper use, and which did the ViT paper adapt?"** All 5 chunks come from the ViT paper, which discusses Adam at length, so the Transformer paper's single sentence "We used the Adam optimizer" never makes the top 5. Ranking chunks one by one can't guarantee that two documents are both represented.

### The "not in the documents" gate (`MIN_SCORE = 0.45`)

Best dense cosine per question at 384 tokens:

| Kind | n | Lowest | Median | Highest |
|---|---|---|---|---|
| answerable | 55 | 0.532 | 0.684 | 0.825 |
| unanswerable (on-topic) | 8 | 0.406 | 0.638 | 0.726 |
| off-topic | 15 | 0.349 | 0.395 | 0.445 |

At 0.45: **0 answerable questions rejected, 0 of 15 off-topic accepted**, the same at every chunk size. One unanswerable question ("How do I install Node.js on Windows?") is stopped by the gate, which is the right outcome by a cheaper route. Phase 4's calibration found a real syllabus question at 0.465 and off-topic questions up to 0.435. Both data sets put the gap between about 0.445 and 0.465, and **0.45 sits in it, so it stays.** The margin on the off-topic side is small (0.005), and a borderline off-topic question that slips through costs one LLM call that answers "not in the documents".

**How many passages to give the LLM (k):** with the reranker, R@3 0.93, R@5 0.97, R@8 0.98. Gains stop at 5, and 8 would add 60% more text to every prompt for one question. **k = 5 stays.**

## Step 5 — Answer benchmark (`python -m eval.answers`)

The whole pipeline: retrieval → Gemini → Phase 6 checks, on all 78 questions. Each answer is saved to `eval/runs/answers-<tag>.jsonl` as it arrives, so a rate limit only pauses a run, and saved answers can be re-scored without new LLM calls.

**Model: `gemini-3.5-flash-lite`, pinned for both runs.** The default `gemini-3.6-flash` allows only **20 requests a day** on the free tier (the API's 429 reports `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota 20), and it was used up before the first question. Without pinning, the app's fallback would have mixed two models within a run. `--no-fallback` makes a run wait and resume instead of switching. This is also the model the app falls back to after 20 questions a day, so it's the one that answers most questions in practice.

**Scoring** (`eval/metrics.py: answer_scores`)
- **correct**: every fact group is in the answer, and no forbidden fact. A `multi` question must also cite every document it's about. multi-01 (Adam / Adafactor) was answered entirely from the ViT paper, which mentions Adam too, so it counts as partial.
- **partial**: some facts.
- **refused**: no facts, and the answer says the documents don't give them, even if it cites related passages ("the exact ImageNet-v2 number is not stated").
- **wrong**: everything else.
- Citations: **cites evidence** means a cited passage contains an evidence quote. **Quote on evidence** means the highlighted quote itself overlaps it, i.e. the highlight lands in the right place.

### Results: current default vs. reranker

62 LLM calls per run (16 questions stopped by the gate), 7 s apart.

| | Hybrid (old default) | **Reranker, top 10 (new default)** |
|---|---|---|
| Syllabus (16) | 16 correct | 16 correct |
| Attention paper (18) | 16 correct, 2 refused | 17 correct, 1 refused |
| ViT paper (18) | 17 correct, 1 refused | 18 correct |
| Both papers (3) | 1 correct, 2 partial | 2 correct, 1 partial |
| **All answerable (55)** | **50 correct**, 2 partial, 3 refused, **0 wrong** | **53 correct**, 1 partial, 1 refused, **0 wrong** |
| A cited passage holds the evidence | 50 | 54 |
| The highlighted quote is on the evidence | 46 | 48 |
| Citations to the wrong document | 0 | 0 |
| Unanswerable (8): "not in the documents" | 8 (1 by the gate) | 8 (1 by the gate) |
| Off-topic (15): "not in the documents" | 15 (all by the gate) | 15 (all by the gate) |
| Median time per answered question | 2.1 s | 2.5 s |

**Findings**
- **No wrong answers in 124.** When the evidence wasn't among the 5 passages, Gemini said so instead of guessing. Every non-correct answer is a retrieval miss.
- **The reranker fixed 3 of those misses and broke nothing:** who proposed multi-head attention (the author-contributions footnote), the ImageNet-v2 number in Table 1, and the cross-paper hardware question. Three wins out of 55 isn't significant on its own, but it matches the retrieval result (p = 0.0005), and each fixed question is one retrieval predicted.
- **Still failing with the reranker:** the parameter count at the end of a table row (no method retrieves it), and multi-01, where one paper crowds the other out of the top 5 (Step 4).
- **The Phase 6 checks warned on 18 of the hybrid run's 50 correct answers, and on only 1 of the 5 that weren't fully correct.** With nothing invented, there was little for them to catch, so nearly every warning was a false alarm. Step 6 fixes that.

**Decision: reranker on by default (`USE_RERANKER = 1`), reranking the top 10 (`RETRIEVE_CANDIDATES = 10`).** The cost is about 0.4 s per question and one more local model (`Alibaba-NLP/gte-reranker-modernbert-base`, already downloaded in Phase 4). `app.py` now loads it at startup with the other models. `CITERAG_USE_RERANKER=0` turns it off.

## Step 6 — The citation checks on real answers

### What was wrong

In the hybrid run, 22 of 79 checked statements were warned. Reading each against the documents, all 22 are true, and one of them cites a weak passage (Gemini cited the sentence listing the reciprocal square-root schedule as an option, not the one where the authors choose it). Causes:

| Cause | Warnings | Example |
|---|---|---|
| Tables and bare list items | 7 | "Width: 1664", "Internal evaluation: 20 marks" |
| Lead-in phrases | 6 | "**Based on the provided excerpts,** the error rate … does not improve past 27%" |
| LaTeX in the answer | 4 | `$\beta_1 = 0.9$`, where the PDF text says "β1 = 0.9" (and "beta_1" read as a missing name) |
| The answer repeats the question's subject | 3 | "The pass marks for the **Web Technology** exam are 45": the exam section never names the course |
| Paraphrase, "This …" referring back, two halves citing different passages | 3 | "This mechanism is similar to … SVMs" |
| Number formatting | 1 | "10,000" vs "10 000" in the PDF |

The second run (reranker) added one more pattern: **lists of bare terms.** In "jQuery covers: Basics / Selectors / Attributes / …", every item was warned, which turns the whole list yellow. A single word isn't a sentence an NLI model can judge, and topic lists are what syllabus questions produce.

**What the NLI model actually reacts to** (same passage, one change at a time):

| Statement | P(entailment) |
|---|---|
| Based on the provided excerpts, the error rate of the L/16 model … does not improve past 27%. | 0.021 |
| The error rate of the L/16 model … does not improve past 27%. | **0.995** |
| The **base Transformer** encoder is composed of a stack of N = 6 identical layers. | 0.005 |
| The encoder is composed of a stack of N = 6 identical layers. | **0.994** |
| … with the question in front of the passage: "The base Transformer encoder … 6 …" | **0.958** |
| … same, with a tampered number: "… 8 identical layers" | 0.001 (contradiction 1.00) |

LaTeX vs. plain made no difference to the NLI model (0.006 vs 0.005), only to the name check.

### What changed (`citerag/validator.py`)

1. **Lead-ins are removed before NLI**: "Based on / According to the provided documents/excerpts/sources…", "In the provided excerpts,", and a leading "Yes,". "No," stays, because it carries the claim.
2. **The question goes in front of each passage** given to the NLI model. It says what the passage is about ("Transformer", "Web Technology"), which the passage itself often doesn't.
3. **A name the question introduces only has to appear in some retrieved passage**, not in the cited one. This keeps rule 2 safe: in a false-premise question ("pass marks for the **Data Science** exam?"), the question context makes NLI accept "The pass marks for the Data Science exam are 45", but no retrieved passage mentions Data Science, so the name check still warns. Tested both ways.
4. **LaTeX becomes plain text** in checked statements and in the Checks list (`\beta_1` → β1, `\frac{1}{\sqrt{d_k}}` → 1/√dk). "10 000" in a passage now matches "10,000".
5. **Bare terms** (3 content words or fewer, no digits, no negation) skip NLI. They pass if what they cite contains them: "Selectors" passes, "Widgets" warns. Short items with numbers ("Pass marks: 100") or negations ("Not mandatory") still go to NLI, since a table label could be paired with the wrong value.
6. "The excerpts do not **name/identify** …" counts as a "not in the documents" statement (no citation needed).

### How it was measured

- `python -m eval.answers --tag … --recheck` rebuilds each saved answer (chunk texts read back from the eval index) and re-runs the current checker, with no LLM calls. Answers saved before segments were stored are rebuilt by `segments_from_text`, which is tested to give exactly the streaming parser's segments. Re-running the old checker reproduced the live run's 22/79 exactly.
- **False alarms**: warnings on the 170 real statements from both runs.
- **Catches**: in each cited statement, one change that makes it false: a number changed (45 → 46; 94 cases), or a word swapped (Adam → SGD, Rails → Django, P100 → V100, more → less, mandatory → optional, Selectors → Widgets…; 54 cases). Encoder → decoder was dropped from the swaps because the decoder also has 6 layers, so the swap is still true.

| Checker | True statements warned | Altered statements caught | Balanced accuracy |
|---|---|---|---|
| Phase 6 (before) | 48/170 (28%) | 145/148 (98%) | 0.85 |
| **Phase 8** | **13/170 (8%)** | 138/148 (93%) | **0.93** |

**What the new checker misses** (7 more than before, out of 148):
- A swapped name that the cited passage also contains: "removed from **JFT-300M**" (the paper discusses both datasets), and "(**GAP**) head" for MAP. With the question as context, NLI accepts these.
- "English-to-**French** … 28.4": the statement cites two quotes, and the joined premise ("…English-to-French… our big model…" + "…28.4 BLEU…") mixes facts from both. This weakness predates Phase 8.
- Digits inside formulas: `10^{-9}` → `10^{-10}` and `\frac{1}{…}` → `\frac{2}{…}`. The number check compares single numbers, and "10" and "2" occur elsewhere in the passage.

**The 13 remaining warnings:** 3 point at a real problem (the weak citation above; a highlighted quote taken from the wrong table row; "top-1" added to an ImageNet-v2 number the table doesn't call top-1). The other 10 are tables, contrasts ("rather than letting it gradually increase…"), a bibliography entry, and the vit-02 sentence whose two halves cite different passages.

**NLI threshold** (new checker, both runs):

| Threshold | True statements warned | Altered caught | Balanced accuracy |
|---|---|---|---|
| 0.3 | 6/170 | 137/148 | 0.945 |
| 0.4 | 8/170 | 137/148 | 0.939 |
| **0.5** | 13/170 | 138/148 | 0.928 |
| 0.6 | 18/170 | 139/148 | 0.917 |
| 0.7 | 20/170 | 141/148 | 0.917 |

A lower threshold looks slightly better, but the difference is about 5 statements, and most catches come from the number/name checks, which the threshold doesn't affect. Phase 6 calibrated 0.5 on a separate hand-labelled set. **0.5 stays.**

## Decisions

| Setting | Before | After | Evidence |
|---|---|---|---|
| Reranker | off | **on** (`USE_RERANKER=1`) | R@1 0.56 → 0.84 (p = 0.0005); answers 50 → 53/55 correct |
| Candidates per search | 30 | **10** | Same accuracy as 30, 3× faster reranking (0.45 s vs 1.4 s) |
| Chunk size | 384 | 384 | No size better by more than 1 of 55 questions |
| Passages given to the LLM (k) | 5 | 5 | R@5 0.97, R@8 0.98 |
| Relevance gate | 0.45 | 0.45 | 0/55 answerable rejected, 0/15 off-topic accepted |
| NLI threshold | 0.5 | 0.5 | Sweep within ~5 statements |
| Citation checks | Phase 6 | lead-ins, question context, LaTeX, bare terms | True statements warned 28% → 8%, altered caught 98% → 93% |

## Tests

- **`tests/test_eval.py` (20):** finding quotes (case, whitespace, curly quotes), question-set mistakes, coverage only when a chunk holds a whole quote, retrieval scores, gate table, fact matching, outcomes (correct/partial/refused/wrong, unanswerable, forbidden facts, multi-document answers must cite both documents), citation scores, segment rebuilding (4 answer shapes, identical to the streaming parser), the real question set's format, and that every quote is found in the documents (skipped when `data/` doesn't have them).
- **`tests/test_validator.py` (+18):** LaTeX to plain text (5 cases), `warmup_steps` outside math untouched, lead-ins not given to NLI (4 cases), the question given with each passage, names from the question (Web Technology passes, a Data Science false premise still warns), bare list items (Selectors ok, Widgets warns, no NLI call), numbers and negations still going to NLI (3 cases), "10 000" = "10,000", "do not name" as a not-found statement.
- **`tests/test_ui.py`:** the reranker preload draws nothing on the page (checked by putting the bug back: the test fails).

## How to run

```bash
python -m eval.retrieval                                  # current settings: retrieval + gate, ~1 min, free
python -m eval.retrieval --sizes 128 256 384 448          # chunk-size sweep
python -m eval.retrieval --methods hybrid rerank --candidates 10 30
python -m eval.answers --tag mytag                        # full pipeline with the LLM (resumable)
CITERAG_GEMINI_MODEL=gemini-3.5-flash-lite python -m eval.answers --tag lite --no-fallback --delay 7
python -m eval.answers --tag mytag --recheck              # re-run the checks on saved answers, no LLM
```

Indexes are cached in `eval/.cache/`, and run outputs go to `eval/runs/`. Both are git-ignored: run outputs quote the documents, and `data/` is git-ignored too. The question set needs the three PDFs in `data/`.

## Known limitations

- **Small, hand-written question set.** 55 answerable questions, so differences of 1–3 questions are noise; that's why sign tests are reported. They were written by reading the extracted text, which favours keyword search (Step 2).
- **One LLM measured.** `gemini-3.6-flash` couldn't be run (20 requests/day on the free tier), and Claude needs a paid key.
- **Fact matching is string-based.** A correct string inside a wrong sentence would count as correct. About a dozen answers were read in full and every warned statement was checked by hand, but not every answer.
- **Tables:** values at the end of a table row are hard to retrieve (att-17) and hard to check (most of the remaining false alarms).
- **Questions about two documents:** ranking chunks one by one can't guarantee both documents reach the top 5 (multi-01).
- **Checker:** the misses listed in Step 6.
