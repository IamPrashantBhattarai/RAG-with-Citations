"""Streamlit UI for CiteRAG. Run with: streamlit run app.py  (Phase 7)

Ask questions about your documents. The answer streams in with [n]
citations; each source opens to its passage (quote highlighted) and the
cited PDF page, and the Phase 6 checks highlight statements their sources
don't support. Documents are added, removed and rescanned in the sidebar.

Sign-in (Phase 13): with Google sign-in set up in .streamlit/secrets.toml,
only the people in access.toml get in, each sees only their groups'
documents (folders under data/), and only admins manage documents. Without
it, the app is a local single-user tool, as before.

Limits (Phase 16): with sign-in, everyone but admins has a number of
questions a day and a minute (access.toml [limits]), and anyone in
`blocked` is shut out.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import streamlit as st

from citerag import auth, config, privacy, ui, web
from citerag.access import (LOCAL_USER, PUBLIC, AccessError, Policy, Registrations, ScopedRetriever, User, group_of,
                            toml_entry)
from citerag.chats import Chat, ChatStore, Turn
from citerag.conversation import is_follow_up, standalone_question
from citerag.generator import Answer, GenerationError, Generator, make_backend, pages_label
from citerag.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_dir, load_file
from citerag.retriever import Retriever
from citerag.screen import default_screener
from citerag.store import EmbeddingMismatchError, Store
from citerag.usage import NO_LIMIT, Allowance, Quota, Usage, UsageError
from citerag.validator import Validator

PROVIDERS = {"gemini": "Gemini (free tier)", "anthropic": "Claude"}
ROOT_FOLDER = "(no folder: admins only)"

# .streamlit/config.toml turns Streamlit's module watcher off, but Streamlit
# restarts it when a browser session reconnects. It then logs a traceback for
# every transformers module that needs torchvision (not installed, not used).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="CiteRAG")


@st.cache_resource(show_spinner="Loading the index and models (first run only)…")
def pipeline():
    store = Store(screener=default_screener())  # new and changed files are screened for prompt injection (Phase 15)
    retriever = Retriever(store)
    validator = Validator()
    # Load the models now, not during the first question. Assigned, because
    # Streamlit "magic" would draw a bare expression on the page.
    _ = store.embedder.model, validator.nli.model
    if retriever.reranker:
        _ = retriever.reranker.model
    return store, retriever, validator


@st.cache_resource
def backend(provider: str):
    return make_backend(provider)


@st.cache_resource
def chat_store(folder: str) -> ChatStore:
    return ChatStore(Path(folder))


@st.cache_resource
def _policy(path: str, mtime: float) -> Policy:  # mtime: an edited access.toml is read again
    return Policy.load(Path(path))


def policy() -> Policy:
    path = config.ACCESS_FILE
    return _policy(str(path), path.stat().st_mtime if path.exists() else 0.0)


def open_chat(chat: Chat) -> None:
    """Show `chat`, and put its id in the address so a reload comes back to it."""
    st.session_state.chat = chat
    if chat.turns:
        st.query_params["chat"] = chat.id
    else:
        st.query_params.pop("chat", None)


@st.cache_data(max_entries=32, show_spinner=False)
def cited_pages(path: str, mtime: float, first: int, last: int) -> bytes:
    return ui.page_pdf(Path(path), first, last)  # mtime: a changed file isn't served from cache


def flash(kind: str, message: str) -> None:
    """Show a message after the next rerun."""
    st.session_state.flash.append((kind, message))


# -- sign-in -------------------------------------------------------------------------


def current_user() -> tuple[User, Quota]:
    """Who is using the app, and how many questions they may ask.

    Stops the page with a sign-in, "no access" or "access removed" screen if nobody may use it.
    """
    try:
        signing_in = auth.configured()
    except auth.AuthConfigError as e:  # broken, not absent: stop rather than open without sign-in
        st.title("CiteRAG")
        st.error(f"The sign-in settings have a problem, so the app won't open: {e}. "
                 "Fix `.streamlit/secrets.toml`, then restart the app.")
        st.stop()
    if not signing_in:
        return LOCAL_USER, NO_LIMIT
    email = auth.identity()
    if email is None:
        st.title("CiteRAG")
        st.caption("Ask questions about your documents. Answers cite the passages they come from.")
        st.button("Sign in with Google", type="primary", on_click=auth.sign_in)
        st.stop()
    try:
        pol = policy()
    except AccessError as e:
        st.title("CiteRAG")
        st.error(f"Sign-in is set up, but the access list has a problem: {e}")
        st.button("Sign out", on_click=auth.sign_out)
        st.stop()
    if pol.is_blocked(email):  # checked on every click, so a block applies at once (Phase 16)
        st.title("CiteRAG")
        st.error("Your access to CiteRAG has been removed. If you think that's a mistake, contact the admin.")
        st.button("Sign out", on_click=auth.sign_out)
        st.stop()
    user = pol.resolve(email)  # with open registration, an unlisted email is a public user
    if user is None:
        st.title("CiteRAG")
        st.error(f"{email} isn't on the access list. Ask an admin to add you.")
        st.button("Sign out", on_click=auth.sign_out)
        st.stop()
    if st.session_state.get("signed_in_as") != email:  # once per browser session, not on every rerun
        Registrations().seen(email)
        st.session_state.signed_in_as = email
    return user, pol.quota(user)


def masker(user: User):
    """How private values are shown to this user: masked for signed-in non-admins (Phase 15)."""
    return privacy.mask if config.MASK_PRIVATE and not user.admin else None


def held_back_passages(flagged: list) -> None:
    """Admins: the passages kept from the model because they look like instructions to it (Phase 15)."""
    n = len(flagged)
    st.warning(f"{n} passage{'s' * (n != 1)} held back from answers: {'they look' if n != 1 else 'it looks'} "
               "like instructions to the AI. Check the file; if it's a false alarm, the passage can be reworded.")
    with st.expander("Show held-back passages"):
        for h in flagged:
            pages = pages_label(h.source, h.page_start, h.page_end)
            st.markdown(f"**{ui.escape_md(h.source)}**" + (f", {pages}" if pages else "")
                        + f"  \n:gray[{ui.escape_md(h.screen)}]")
            st.text(h.text if len(h.text) <= 800 else h.text[:800] + " …")


def waiting_users() -> None:
    """Admins: people who signed up through open registration and are still public users."""
    try:
        pol = policy()
    except AccessError:
        return
    waiting = Registrations().unlisted(pol)
    if not waiting:
        return
    with st.expander(f"Registered, public only ({len(waiting)})"):
        st.caption("They can read the public documents. To give someone a group, paste their block into "
                   "access.toml and fill in the groups; it applies on their next page load.")
        for email, seen in waiting:
            st.markdown(f"{ui.escape_md(email)}  \n:gray[first signed in {seen.get('first', '?')[:10]}, "
                        f"last {seen.get('last', '?')[:10]}]")
            st.code(toml_entry(email), language="toml")
    if not pol.open_registration:
        st.caption("Open registration is off, so they can't sign in until they're listed.")


def questions_today() -> None:
    """Admins: how many questions each person has asked today, busiest first (Phase 16)."""
    try:
        pol = policy()
        usage = Usage()
        rows, shared = usage.today(), usage.shared_today()
    except (AccessError, UsageError):
        st.caption("Today's question counts can't be read.")
        return
    with st.expander(f"Questions today ({sum(n for _, n in rows)})"):  # shown even at 0, so admins know it's there
        if not rows:
            st.caption("Nobody has asked a question yet today.")
        elif pol.open_registration:
            st.caption(f"Sign-ups together: {shared} of {pol.limits.public_total}.")
        for email, n in rows:
            if pol.is_blocked(email):
                note = f"{n}, blocked"
            else:
                daily = pol.quota(pol.user(email) or User(email, listed=False)).daily
                note = f"{n}, no limit" if daily is None else f"{n} of {daily}"
            st.markdown(f"{ui.escape_md(email)}  \n:gray[{note}]")
        st.caption("Limits are in access.toml: [limits] for everyone, daily_questions for one person, "
                   "blocked to shut someone out.")


def show_left(slot, allowance: Allowance) -> None:
    """The "questions left today" line under the signed-in name."""
    if slot is not None and allowance.daily is not None and allowance.left is not None:
        slot.caption(f"{allowance.left} of {allowance.daily} question{'s' * (allowance.daily != 1)} left today")


def chats_folder(user: User) -> str:
    # The local user keeps the chats saved before sign-in existed; each signed-in user has their own folder.
    return str(config.CHATS_DIR if user is LOCAL_USER else config.CHATS_DIR / user.key)


# -- sidebar -------------------------------------------------------------------------


def sidebar(store: Store, user: User, allowance: Allowance) -> dict:
    left_slot = None  # "questions left today": filled below, and updated after a question
    with st.sidebar:
        if user is not LOCAL_USER:
            role = " · admin" if user.admin else "" if user.listed else " · public documents only"
            st.caption(f"Signed in as {ui.escape_md(user.email)}{role}")
            left_slot = st.empty()
            show_left(left_slot, allowance)
            if not user.listed:
                st.caption("You can read the public documents. An admin can add you to a group for more.")
            st.button("Sign out", on_click=auth.sign_out, width="stretch")
            st.divider()
        st.subheader("Documents")
        docs = {name: n for name, n in store.sources().items() if user.can_read(name)}
        flagged = store.quarantined() if user.admin else []
        held = {name: sum(h.source == name for h in flagged) for name in docs}
        for name, chunks in docs.items():
            left, right = st.columns([6, 1], vertical_alignment="center")
            note = f" · {held[name]} held back" if held.get(name) else ""
            left.markdown(f"{ui.escape_md(name)}  \n:gray[{chunks} chunk{'s' * (chunks != 1)}{note}]")
            if user.admin and right.button("", key=f"remove-{name}", icon=":material/close:", type="tertiary",
                                           help=f"Remove from the index (the file stays in {config.DATA_DIR.name}/)"):
                store.delete_source(name)
                flash("success", f"Removed {name} from the index.")
                st.rerun()
        if not docs:
            st.caption("No documents indexed yet." if user.admin else "No documents are shared with you yet.")
        if flagged:
            held_back_passages(flagged)
        if user.admin and user is not LOCAL_USER:
            waiting_users()
            questions_today()
        if user.admin:
            manage_documents(store, user)

        st.divider()
        st.subheader("Settings")
        names = list(PROVIDERS)
        provider = st.radio("Answer with", names, format_func=PROVIDERS.get,
                            index=names.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in names else 0)
        if not ui.has_key(provider):
            key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
            st.warning(f"No API key found: set {key} in .env and restart.")
        if provider == "gemini":
            st.caption("Free tier: Google may use prompts and answers to improve its products. "
                       "Don't use it with confidential documents.")
        sources = st.multiselect("Search in", list(docs), placeholder="All documents")
        k = st.slider("Passages given to the model", 1, 10, config.TOP_K)
        validate = st.toggle("Check citations", value=config.VALIDATE,
                             help="Check every statement against the passage it cites.")
        strict = st.toggle("Strict mode", value=config.STRICT, disabled=not validate,
                           help="Remove statements that fail the checks. The answer appears once it "
                                "has been checked instead of streaming.")
        chats_section(user)
    return {"provider": provider, "sources": sources or None, "k": k, "validate": validate,
            "strict": strict and validate, "documents": list(docs), "left_slot": left_slot}


def manage_documents(store: Store, user: User) -> None:
    """Uploads, web pages and rescans: admins only."""
    folder = config.DATA_DIR
    if user is not LOCAL_USER:  # with sign-in, a document goes into a group's folder
        groups = sorted({g for g in (group_of(s) for s in store.sources()) if g}
                        | {p.name for p in config.DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")}
                        | {PUBLIC})
        choice = st.selectbox("Into folder (group)", groups + [ROOT_FOLDER], index=groups.index(PUBLIC),
                              help="Folders are groups: people see the folders of their groups in "
                                   "access.toml. Everyone signed in sees public.")
        folder = config.DATA_DIR if choice == ROOT_FOLDER else config.DATA_DIR / choice
    files = st.file_uploader("Add documents", type=sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
                             accept_multiple_files=True, key=f"upload-{st.session_state.upload_key}")
    if files and st.button("Add to index", type="primary", width="stretch"):
        add_files(store, files, folder)
    with st.form("add-url", clear_on_submit=True, border=False):
        url = st.text_input("Add a web page", placeholder="https://…",
                            help="The page is saved into the documents folder and indexed. "
                                 "Only public http(s) pages, up to 5 MB.")
        if st.form_submit_button("Save and index", width="stretch") and url.strip():
            add_url(store, url, folder)
    if st.button(f"Rescan {config.DATA_DIR.name}/ folder", width="stretch",
                 help=f"Index new or changed files in {config.DATA_DIR.name}/ and drop files that are gone"):
        rescan(store)


def chats_section(user: User) -> None:
    st.divider()
    st.subheader("Chats")
    store = chat_store(chats_folder(user))
    current = st.session_state.chat
    if st.button("New chat", width="stretch", disabled=not current.turns):
        open_chat(Chat())
        st.rerun()
    for chat_id, title in store.list():
        if st.button(title, key=f"chat-{chat_id}", type="tertiary", disabled=chat_id == current.id):
            open_chat(store.load(chat_id) or Chat())
            st.rerun()
    if current.turns and st.button("Delete this chat", width="stretch"):
        store.delete(current.id)
        open_chat(Chat())
        st.rerun()


def source_name(path: Path) -> str:
    """How a saved file is named in the index: its path under data/ ("hr/policy.pdf")."""
    return path.resolve().relative_to(config.DATA_DIR.resolve()).as_posix()


def add_files(store: Store, files, folder: Path) -> None:
    pages, errors = [], []
    with st.spinner("Reading and indexing…"):
        for f in files:
            try:
                path = ui.save_upload(f.name, f.getvalue(), folder)
                loaded = load_file(path, source_name(path))
            except (ValueError, LoaderError) as e:
                errors.append(str(e))
                continue
            if not loaded:
                errors.append(f"{path.name}: no text found, even with OCR")
            pages.extend(loaded)
        if pages:
            flash("success", "Indexed. " + ui.report_text(store.index(pages)))
    for e in errors:
        flash("error", e)
    st.session_state.upload_key += 1  # empties the uploader
    st.rerun()


def add_url(store: Store, url: str, folder: Path) -> None:
    with st.spinner("Downloading and indexing…"):
        try:
            path = web.fetch(url, folder)
            pages = load_file(path, source_name(path))
        except (web.WebError, LoaderError) as e:
            flash("error", str(e))
            st.rerun()
        if pages:
            flash("success", f"Saved {source_name(path)}. " + ui.report_text(store.index(pages)))
        else:
            flash("error", f"{path.name}: no text found on the page")
    st.rerun()


def rescan(store: Store) -> None:
    with st.spinner("Scanning…"):
        try:
            report = store.index(load_dir(config.DATA_DIR), prune=True)
            flash("success", "Rescanned. " + ui.report_text(report))
        except LoaderError as e:
            flash("error", str(e))
    st.rerun()


# -- answers -------------------------------------------------------------------------


def ask(question: str, settings: dict, retriever, validator: Validator, mask=None) -> Answer | None:
    generator = Generator(retriever, backend(settings["provider"]), validator,
                          validate=settings["validate"], strict=settings["strict"])
    placeholder = st.empty()
    placeholder.caption("Searching…")
    shown: list[str] = []

    def on_text(piece: str) -> None:
        shown.append(piece)
        text = "".join(shown)
        if mask:  # a number still arriving can't be recognised yet, so it waits for the next piece
            text = mask(privacy.streaming_safe(text))
        placeholder.markdown(ui.streaming_markdown(text) + " ▌")

    wait = st.spinner("Writing and checking the answer…") if settings["strict"] else nullcontext()
    try:
        with wait:
            answer = generator.answer(question, settings["k"], settings["sources"],
                                      on_text=None if settings["strict"] else on_text)
    except GenerationError as e:
        placeholder.empty()
        st.error(str(e))
        return None
    placeholder.empty()
    return answer


def take_question(question: str, user: User, quota: Quota, usage: Usage | None, left_slot) -> Allowance | None:
    """Count the question against the user's limits (Phase 16); None, with the reason shown, if it can't be asked."""
    if len(question) > config.MAX_QUESTION_CHARS:  # the box stops typing there; this catches anything sent past it
        st.warning(f"That question is too long: keep it to {config.MAX_QUESTION_CHARS:,} characters or fewer.")
        return None
    taken = usage.take(user.email, quota) if usage else Allowance(True)
    if not taken.allowed:
        st.warning(taken.message)
        return None
    show_left(left_slot, taken)
    return taken


def show_answer(answer: Answer, turn: int, user: User) -> None:
    mask = masker(user)
    st.markdown(ui.answer_markdown(answer, mask=mask), unsafe_allow_html=True)
    if line := ui.summary_line(answer):
        st.caption(line)
    if found := privacy.find_private(answer.text):
        how = ", masked" if mask else ", masked for members" if user is not LOCAL_USER and config.MASK_PRIVATE else ""
        st.caption(f"This answer contains private data ({privacy.summary(found)}){how}.")
    if answer.held_back:
        places = sorted({ui.escape_md(h["source"]) + (f", {p}" if (p := pages_label(h["source"], h["page_start"], h["page_end"])) else "")
                         for h in answer.held_back})
        n = len(answer.held_back)
        st.caption(f"{n} passage{'s' * (n != 1)} held back from the model because {'they look' if n != 1 else 'it looks'} "
                   f"like instructions to the AI: {'; '.join(places)}.")

    hits = {h.chunk_id: h for h in answer.hits}
    for c in answer.citations:
        with st.expander(f"\\[{c.number}\\] {ui.escape_md(c.source)}" + (f", {c.pages}" if c.pages else "")):
            if c.section:
                st.caption(f"Section: {ui.escape_md(c.section)}")
            if c.chunk_id in hits:
                st.markdown(ui.passage_html(hits[c.chunk_id].text, c.start_char, c.end_char, mask=mask),
                            unsafe_allow_html=True)
            if not user.can_read(c.source):
                continue
            pdf = ui.source_pdf(c.source, config.DATA_DIR)
            if pdf is not None and st.toggle(f"Show PDF {c.pages}", key=f"pdf-{turn}-{c.number}"):
                st.pdf(cited_pages(str(pdf), pdf.stat().st_mtime, c.page_start, c.page_end), height=600)
            image = ui.source_image(c.source, config.DATA_DIR)
            if image is not None and st.toggle("Show image", key=f"img-{turn}-{c.number}"):
                st.image(str(image))

    if answer.validation is not None and (lines := ui.check_lines(answer.validation, mask=mask)):
        with st.expander("Checks"):
            st.markdown("\n\n".join(lines))
            if answer.validation.invalid_markers:
                st.caption("Dropped markers pointing at sources that don't exist: "
                           + ", ".join(f"\\[{n}\\]" for n in answer.validation.invalid_markers))
    if answer.hits:
        with st.expander("Retrieved passages"):
            if answer.model:
                model = f"Model: {answer.model}"
                if answer.usage:
                    model += f" · {answer.usage['input_tokens']} in / {answer.usage['output_tokens']} out tokens"
                st.caption(model)
            cited = {c.chunk_id for c in answer.citations}
            lines = []
            for n, h in enumerate(answer.hits, 1):
                pages = pages_label(h.source, h.page_start, h.page_end)
                where = ui.escape_md(h.source) + (f", {pages}" if pages else "") + (f" · {ui.escape_md(h.section)}" if h.section else "")
                scores = " · ".join(f"{name} {value:.3f}" for name, value in h.scores.items())
                lines.append(f"{n}. {where} · {'**cited**' if h.chunk_id in cited else 'not cited'}  \n"
                             f"   :gray[{scores}]")
            st.markdown("\n".join(lines))


# -- page ----------------------------------------------------------------------------

user, quota = current_user()  # stops here with a sign-in screen when sign-in is set up and nobody is signed in
usage = None if user is LOCAL_USER else Usage()  # the local user is never counted: there's nobody to limit

if "chat" not in st.session_state:  # first run of this browser session: reopen the chat in the address
    st.session_state.chat = chat_store(chats_folder(user)).load(st.query_params.get("chat", "")) or Chat()
st.session_state.setdefault("flash", [])
st.session_state.setdefault("upload_key", 0)

st.title("CiteRAG")
st.caption("Ask questions about your documents. Answers cite the passages they come from.")

try:
    store, retriever, validator = pipeline()
except EmbeddingMismatchError as e:
    st.error(str(e))
    st.stop()

allowance = usage.status(user.email, quota) if usage else Allowance(True)
settings = sidebar(store, user, allowance)
for kind, message in st.session_state.flash:
    getattr(st, kind)(message)
st.session_state.flash = []

documents = settings["documents"]
if not documents:
    st.info("Add a PDF, TXT or MD file in the sidebar to get started." if user.admin
            else "No documents are shared with you yet. Ask an admin to add you to a group.")
# Every search goes through the user's documents only; admins (and the local user) see everything.
scoped = retriever if user.admin else ScopedRetriever(retriever, documents)

chat: Chat = st.session_state.chat
for i, turn in enumerate(chat.turns):
    st.markdown(f"**{ui.escape_md(turn.question)}**")
    if not all(user.can_read(h.source) for h in turn.answer.hits):
        st.caption("This answer used documents you no longer have access to, so it's hidden.")
        st.divider()
        continue
    if turn.search != turn.question:  # a rewritten follow-up can carry a number from an earlier answer
        st.caption(f"Searched for: {ui.escape_md((masker(user) or str)(turn.search))}")
    show_answer(turn.answer, i, user)
    st.divider()

if documents and not allowance.allowed:  # out of questions for today (Phase 16)
    st.info(allowance.message)

if question := st.chat_input("Ask a question about your documents", disabled=not documents or not allowance.allowed,
                             max_chars=config.MAX_QUESTION_CHARS):
    st.markdown(f"**{ui.escape_md(question)}**")
    if taken := take_question(question, user, quota, usage, settings["left_slot"]):
        search = question
        if chat.turns and is_follow_up(question):
            with st.spinner("Reading the follow-up…"):
                search = standalone_question(question, chat.history(), backend(settings["provider"]))
            if search != question:
                st.caption(f"Searched for: {ui.escape_md((masker(user) or str)(search))}")
        answer = ask(search, settings, scoped, validator, mask=masker(user))
        if answer is None and usage:  # failed on our side (the AI service, say): the question doesn't count
            usage.give_back(taken.ticket)
            show_left(settings["left_slot"], usage.status(user.email, quota))
        if answer is not None:
            chat.turns.append(Turn(question, search, answer))
            chat_store(chats_folder(user)).save(chat)
            open_chat(chat)
            show_answer(answer, len(chat.turns) - 1, user)
