# Phase 16 — Question limits and blocking

**Status:** ✅ Built (2026-11-15) · Code: `citerag/usage.py` (new), `citerag/access.py`, `citerag/config.py`, `app.py`, `access.example.toml` · Tests: `test_usage.py` (12, new), +16 `test_access.py`, +9 `test_ui.py`; 389 passing after the GitHub fix

**Outcome in one line:** with sign-in on, everyone but admins has a number of questions a day and a minute, sign-ups share one daily pool, and anyone in `blocked` is shut out on their next click. Counting a question takes about 0.6 ms.

## Goal

Open registration (2026-11-13) lets anyone with a Google account in, which made two gaps urgent:
- **Every question uses the AI budget.** A question makes one call to the AI, or two for a follow-up (the rewrite). With Claude that's billed to the key; with free Gemini the whole app shares 20 requests a day on `gemini-3.6-flash`, so one sign-up could use up everyone's day. Each question also runs search, reranking and the citation checks on this computer.
- **One person couldn't be shut out.** With open registration on, anyone unlisted gets in as a public user; the only lever was turning registration off for everyone.

Asking is the only costly thing non-admins can do (uploads, web pages and rescans are admin-only since Phase 13), so only questions are counted.

## Decisions (agreed, 2026-11-15)

| Question | Choice |
|---|---|
| How to block | **A line in `access.toml`** (`blocked = [...]`), where everything else about access lives. A Block button would have meant the app writing the access file, or a second list somewhere else. |
| What to count | **Questions**, not tokens or money: people understand "5 left today". Cost per person fits the later monitoring step better. |
| The numbers | **20 a day per sign-up, 100 per member, 200 for all sign-ups together, 5 a minute for anyone**; admins have no limit. All changeable in `access.toml`. |

## Step 1 — Blocking (`citerag/access.py`)

- `blocked = ["someone@gmail.com"]` at the top of `access.toml`. Emails are matched without case or spaces, like the user list.
- **Blocking beats everything**, including being listed with groups. `Policy.resolve` returns `None` for a blocked email, so no caller can let one in by mistake. To unblock, delete the line.
- The page checks it before anything else (`current_user`) and shows only "Your access to CiteRAG has been removed" and Sign out. The access list is re-read whenever it changes, so a block applies on the person's next click, with no restart.
- A blocked sign-in isn't recorded, and blocked people don't appear under "Registered, public only".
- **A TOML trap, caught:** a bare line after `[limits]` or `[[users]]` belongs to that section, however it's indented. Writing `blocked = [...]` at the bottom of the file would have silently attached it to the last user. The loader now says "blocked belongs at the top of the file, above [limits] and the first [[users]]" (same for `open_registration`).

## Step 2 — Limits (`access.toml` `[limits]`)

```toml
[limits]
public = 20         # per day, each sign-up not in access.toml
member = 100        # per day, each listed user who isn't an admin
public_total = 200  # per day, all sign-ups together
per_minute = 5      # anyone but admins
```

- `daily_questions = 300` on one `[[users]]` entry gives that person their own daily limit. On an admin it's an error (admins have no limit), so the file doesn't promise something it won't do.
- A missing line takes its default. Every value is a whole number, 0 or more; `per_minute = 0` is refused (to stop someone asking, block them).
- **Why a shared pool for sign-ups:** one person can make several Google accounts. `public_total` caps all of them together. When it runs out, sign-ups see "New sign-ups have used all of today's questions between them", and members are unaffected.
- `Policy.quota(user)` turns this into a `Quota(daily, per_minute, shared)`; admins and the local app without sign-in get `NO_LIMIT`.
- Questions longer than 2,000 characters (`CITERAG_MAX_QUESTION_CHARS`) aren't accepted: the box stops typing there, and the server refuses anything sent past it.

## Step 3 — Counting (`citerag/usage.py`)

- **Counted on the server, by the email Google verified.** The page's memory starts again on every refresh or new tab, so a counter kept there would be useless.
- **SQLite** (`.usage.db`, git-ignored, like `.registrations.json`). It comes with Python and keeps counts correct across threads and processes, which matters later when the app runs on a server with several workers. One table: who, when, and whether the question counts towards the sign-ups' pool. Questions older than two days are deleted, since only today and the last minute are ever counted.
- **A day** runs from midnight to midnight on the computer running the app.
- **When a question counts:** when it's sent, before the follow-up rewrite and the search. If answering fails on our side (the AI service is down, the free tier is busy), it's given back. A question that finds no matching passages still counts: it used search and the checks.
- **The per-minute limit** says how long to wait: "That's 5 questions in a minute. Try again in 30 seconds", worked out from the question that has to fall out of the last 60 seconds.
- **If the counts can't be read or written** (a damaged file, a full disk), limited people are refused with "Questions can't be counted right now, so they're paused" and admins carry on. The limits exist to cap spending, so they fail closed.

### Two questions at the same moment

Each question is counted in one transaction that takes SQLite's write lock (`BEGIN IMMEDIATE`) before counting. Otherwise two tabs could both see "9 of 10 used" and both go ahead.

Tested with 20 threads, each with its own connection, sending a question at the same moment against a limit of 10, with a 10 ms pause between counting and recording to make a race likely:

| Version of the counter | Allowed | Wrongly paused |
|---|---|---|
| As built (write lock before counting) | 10 | 0 (10 runs of 10) |
| No write lock, and no write before counting | 1–2 | 18–19 (SQLite saw the clash and refused) |
| Same, without the 10 ms pause | 4–9 | 11–16 |
| No `BEGIN IMMEDIATE` only | 10 | 0: the clean-up `DELETE` at the start of the transaction already takes the write lock |

In SQLite's WAL mode a race shows up as refused questions, not as questions over the limit: the second writer is told its view is out of date. The pause makes the test fail every time on a broken counter instead of only most times, so the test keeps it. `BEGIN IMMEDIATE` stays, so correctness doesn't hang on the order of the statements.

**Found and fixed:** the first run of the race test against the real counter paused 1 question of 20 with "database is locked". Twenty connections were creating a brand-new database file at once, and switching it to WAL mode needs the file to itself. The table is now set up once per process, under a lock. After the fix, 0 pauses in 10 runs.

### Speed

On this Mac, with 5,000 questions from 200 people already counted that day: counting a question takes **0.58 ms** (median; 95th percentile 0.70 ms), and reading what's left 0.43 ms. The admin's "Questions today" list takes 0.48 ms. The file was 444 KB.

## Step 4 — The page (`app.py`)

- **Users:** "14 of 20 questions left today" under their name, updated as soon as a question is counted. Out of questions: the box is turned off with "You've used today's 20 questions. They reset at midnight." Refusals for too many questions in a minute, or for a question that's too long, show as warnings before any search.
- **Admins:** no limit and no counter line. A "Questions today (N)" list in the sidebar shows each person, busiest first, as "12 of 20", "3, no limit" or "5, blocked", plus "Sign-ups together: 45 of 200" when registration is open.
  - It's always shown, with "Nobody has asked a question yet today" at 0. The first version hid it until someone had asked, and when the user went looking for it in the admin account, it wasn't there.
- **Without sign-in** (the local app) nothing is counted and no file is created.

## Step 5 — Tests

- `test_usage.py` (12): the daily limit counts down and resets at midnight (on a fake clock, in Nepal time), per-person counts, `status` counts nothing, the per-minute wait, the shared pool, give-back, no-limit, a limit of 0, old rows deleted, the same-moment race, waiting milliseconds (not seconds) for another question being counted, and a damaged file (limited people paused, admins carry on).
- `test_access.py` (+16): blocked people kept out even when listed, blocked people not waiting, each kind of user's quota and the defaults, 12 mistakes explained (including `blocked` and `open_registration` placed under a section), and `access.example.toml` loads.
- `test_ui.py` (+9): blocked listed user and blocked sign-up shut out (and not recorded), a member counting down to 0 and the box turned off, the per-minute refusal before any search, sign-ups' smaller limit, a failed answer given back, a question that's too long refused and not counted, and the admin's list (shown even before anyone asks).
  - The page tests count into a temporary file, never the real `.usage.db`.

## Trying it

In the user's `access.toml` (2026-11-15): `blocked = []`, the `[limits]` section with the defaults, and a **trial** `daily_questions = 2` on a member account, to be deleted after trying it:
1. Sign in with that account and ask 2 questions. The sidebar counts down, then the box turns off.
2. As admin, open "Questions today" in the sidebar.
3. Add the public-only account to `blocked`, save, and click anything in its window: "Your access has been removed."

## The first push failed on GitHub (2026-11-15, found 2026-11-16)

The tests passed here but failed on GitHub's Linux machine (run 43697346867). Only the "Run the tests" step failed.

**The real cause (found 2026-11-16).** Two page tests checked that admins see no warning:
- `test_admins_have_no_limit_and_see_questions_today` (Phase 16)
- `test_admins_are_warned_about_held_back_passages_and_members_are_not` (Phase 15)

The sidebar warns "No API key found" when no key is set. The user's `.env` has keys, so locally there was never a warning; GitHub has no `.env`, so there was one. Running the page tests with the keys emptied reproduced both failures.
- Fix: the page-test fixture now sets fake keys (the backends are fakes, so nothing is sent), so the result doesn't depend on whoever runs the tests. A new test checks that the warning does appear when no key is set.
- Earlier phases never hit this: their page tests didn't check for the absence of warnings.

**What was done first, on a wrong guess (2026-11-15).** Kept, because it makes the counter behave the same on every machine:
- **The log couldn't be read.** GitHub shows a job's log only to people signed in, and its log API asks for admin rights.
- **Ruled out here:** the full suite also passed with the clock in UTC and without the 1.2 GB detector model, as on GitHub.
- **The guess:** the 20-questions-at-once test. When the write lock is taken, SQLite's busy timeout retries in steps of a few milliseconds, but only where it was compiled knowing it can sleep for less than a second. Otherwise it sleeps a whole second between tries. Twenty questions retrying once a second, in step, can run past the 10-second wait and be paused. GitHub's Python builds SQLite with its configure script and no flag for this, so the build script didn't settle it.
- **Fix, whatever the build:** `_lock_for_writing` turns SQLite's busy timeout off and does the waiting itself, every 2–20 ms at random so waiting sessions don't retry in step, for up to `LOCK_WAIT_SECONDS` (10).
  - New test: while another connection holds the lock for 0.3 s, a question waits 0.25–0.8 s, not a second or more.
  - 20 at once still let exactly 10 through, with 0 pauses in 5 runs.
- **Seeing failures next time:** the workflow now lists each failed test, with its error, on the commit's checks, where anyone can read them ([14-ci.md](14-ci.md)). That is how the real cause was found: the guess didn't fix the run, and the next run named the two tests.

## Limitations

- Limits count questions, not cost: a long Claude answer with heavy thinking costs more than a short one. Dollar tracking per person belongs with monitoring.
- `public_total` caps the damage from many throwaway accounts but doesn't tell them apart. A determined person could use up the sign-ups' pool for the day; members are unaffected.
- The counts are kept on one computer. Several app servers would need to share one database (the server database item on the production list).
- "Midnight" is the server's. When the app runs on a server in another time zone, people's days reset at the server's midnight.
