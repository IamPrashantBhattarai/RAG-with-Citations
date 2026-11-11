# Phase 13 — Sign-in and permissions

**Status:** ✅ Built (2026-11-11). Google sign-in needs the user's OAuth client (manual step below) · Code: `citerag/auth.py` (new), `citerag/access.py` (new), `citerag/chats.py`, `citerag/config.py`, `app.py`; `access.example.toml`, `eval/permissions.py` (new) · Tests: `tests/test_access.py` (14), +6 `tests/test_ui.py`; 275 passing in total

**Outcome in one line:** with Google sign-in set up, only the people listed in `access.toml` get in, each searches only the documents in their groups' folders (enforced at search time, before anything reaches the LLM), only admins manage documents, and chats are per user. On the real search stack, 504 searches by restricted users returned 0 passages they may not read.

## Goal

First item of the production list: the app had no notion of a user, so anyone who could open it saw every document and every chat. Two separate problems:
1. **Who is this?** (authentication)
2. **What may they see?** (authorisation)

The second is the RAG-specific one. A restricted passage that reaches the LLM even once can leak in the answer, so hiding it in the page isn't enough.

## Decisions (asked, 2026-11-11)

| Question | Choice |
|---|---|
| Sign-in provider | **Google** |
| How permissions are organised | **Folders = groups** (`data/hr/…` → group `hr`; `data/public/…` → everyone signed in) |
| Who may sign in | **Only emails listed in `access.toml`** |

## Step 1 — Tech stack

| Part | Choice | Why |
|---|---|---|
| Sign-in | Streamlit's built-in `st.login()` / `st.user` / `st.logout()` (OpenID Connect) | Already in Streamlit 1.66; no passwords stored by CiteRAG; Google handles passwords, 2-factor and recovery |
| Package | `Authlib` 1.9.4 (+ `joserfc` 1.8.0), via `streamlit[pdf,auth]` | The only new install, resolved with `--exclude-newer 2026-11-04` |
| Sign-in settings | `.streamlit/secrets.toml` `[auth]` (git-ignored) | Streamlit's standard place. The template has the redirect URI, Google's metadata URL and a **random 64-character cookie secret** filled in; only the client id and secret are left |
| Who may do what | `access.toml` (git-ignored; `access.example.toml` committed) | Small, readable and reviewable. A database or admin page can come with the API step |
| Enforcement | `access.ScopedRetriever` around the app's retriever | One place where every search is filtered, testable on its own |

## Step 2 — How it works

**`citerag/auth.py`** only answers "which verified email is signed in?":
- `configured()` is true only when all five `[auth]` settings are present and the client id and secret aren't the template's `PASTE…` placeholders. Until then the app stays a local, single-user tool exactly as before.
- `identity()` returns the email only if Google says it is verified.

**`citerag/access.py`** decides what that email may do:
- `Policy.load()` reads `access.toml` and rejects mistakes with a message saying what to fix: unknown keys, bad emails, group names that aren't folder names (so no `../`), a non-boolean `admin`, duplicates.
- Emails are matched without case.
- `User.can_read(source)`:
  - an admin reads everything
  - everyone reads `public/…`
  - a member reads their groups' folders
  - **a file directly in `data/`, in no folder, is admin-only**, because nobody decided who should see it
- `ScopedRetriever(retriever, allowed)` wraps every search. It gives the retriever exactly the user's documents (intersected with "Search in" when that is used), and **checks every hit again afterwards**, logging an error if one slipped through.
- **An empty allowed set returns nothing without searching.** This matters because the underlying retriever (Phase 4) reads an empty or missing source list as "all documents". A user with no groups would otherwise have searched everything.

**App (`app.py`):**
- **Sign-in screen:** with sign-in on and nobody signed in, the page shows only "Sign in with Google" and stops, before the index or models load. A listed user who isn't in `access.toml` gets "isn't on the access list" and a sign-out button. A broken `access.toml` gets an error naming the problem.
- **Sidebar:** shows "Signed in as …" with *Sign out*. The document list and "Search in" show only readable documents. **Uploading, adding web pages, removing and rescanning are admins only.** Admins choose the folder (group) a new document goes into.
- **Uploads keep their folder in the index** (`hr/policy.pdf`, not `policy.pdf`). Before this, `load_file(path)` named an upload by its bare file name, which would have put every upload in no group.
- **Search:** every question goes through `ScopedRetriever`, except for admins and the local user, who read everything. "Show PDF" and "Show image" check `can_read` too.
- **Chats are per user**, in `.chats/<16-hex key of the email>/`; the local user keeps the old `.chats/`. A chat id in the URL is looked up only in the signed-in user's own folder. A saved answer that used a document the user can no longer read is hidden ("This answer used documents you no longer have access to").
- `access.toml` is re-read when it changes (cached by its modification time), so adding someone needs no restart.

## Step 3 — What the user configures (manual)

These are the steps that can't be done from here, because they need the user's Google account:

1. **Google Cloud Console → APIs & Services → OAuth consent screen.** Choose *Internal* (Google Workspace) or *External* (personal accounts; add each person as a test user while the app is in testing). Scopes: `openid`, `email`, `profile`.
2. **Credentials → Create credentials → OAuth client ID → Web application.** Authorised redirect URI: `http://localhost:8501/oauth2callback` (add the server's `https://…/oauth2callback` when it's deployed).
3. **Paste the Client ID and Client Secret** into `.streamlit/secrets.toml` (`client_id`, `client_secret`), then restart the app.
4. **Edit `access.toml`**: who may sign in, their groups, who is admin. The user's own email is already listed as admin.
5. **Sort `data/` into group folders** when ready (`data/public/`, `data/hr/`…). Until then the existing files are in no folder, so admins only, and the admin sees everything as before. Moving files renames their sources (the folder becomes part of the name), so **Rescan** afterwards.

## Step 4 — Evaluation: does anything leak?

`python -m eval.permissions`: all 126 eval questions, searched as four restricted users through the app's real search (dense + keyword + reranker) inside `ScopedRetriever`, over the 8 evaluation documents. No LLM.

| User may read | Passages from other documents | Questions about documents they can't read | …that found nothing |
|---|---|---|---|
| The two papers | **0** | 59 | 44 |
| The company files (.docx, .pptx, .eml) | **0** | 73 | 64 |
| The syllabus | **0** | 82 | 70 |
| Nothing | **0** | 98 | 98 |

**504 searches, 0 leaks.** When a question is about documents a user can't read, search usually finds nothing; the relevance gate then answers "not in the documents" without an LLM call. The other cases return unrelated passages from the user's own documents, which the model answers from or declines.

## Step 5 — Tests

**`tests/test_access.py` (14):**
- listed users and their groups, with case-insensitive emails
- who reads what: public, own groups, admin, files in no folder
- eight kinds of mistakes in `access.toml`, each explained
- a missing file says to copy the example
- `ScopedRetriever`:
  - a retriever that ignores the filter still leaks nothing
  - "Search in" only narrows
  - nothing allowed means nothing searched, and the retriever is never called with an empty list
- per-user chat keys

**`tests/test_ui.py` (+6, headless app, sign-in faked):**
- nobody signed in → only the sign-in button, and nothing searched
- an unlisted account → turned away
- a member:
  - sees only hr and public in the sidebar, with no document management
  - their searches go to `["hr/policy.pdf", "public/guide.pdf"]` only
  - a passage the fake retriever returned from another document never reaches the page
- an admin → sees everything and chooses the upload folder
- chats are saved in the user's own folder, and another user doesn't see them
- a broken `access.toml` → reported

The existing app tests now pin sign-in off, so they keep testing the local mode whatever `secrets.toml` holds.

## Fixed along the way

- Chats saved in the same second sorted at random (timestamps had second resolution). Now microseconds. The test caught it once in three runs.

## Limitations

- **Sign-in isn't tested against Google here.** That needs the user's OAuth client (Step 3). The flow is Streamlit's own; CiteRAG's side is tested with a faked identity.
- **`localhost` only, for now.** Google allows `http://localhost` redirects for testing; a shared server needs HTTPS and its own redirect URI (the API/Docker step).
- **No audit log yet** of who asked what or which documents were used.
- **Permissions are per folder, not per file;** a file's group changes when it moves (then Rescan).
- **Admins are all-or-nothing:** there's no "can upload to hr only" role yet.
- **The Streamlit server holds everything in one process:** all users share the index. Permissions are enforced in the app, so the app process itself must be trusted (no untrusted code runs in it).
