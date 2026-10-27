"""Streamlit UI for CiteRAG. Run with: streamlit run app.py  (Phase 7)

Ask questions about your documents. The answer streams in with [n]
citations; each source opens to its passage (quote highlighted) and the
cited PDF page, and the Phase 6 checks highlight statements their sources
don't support. Documents are added, removed and rescanned in the sidebar.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import streamlit as st

from citerag import config, ui
from citerag.generator import Answer, GenerationError, Generator, make_backend, page_label
from citerag.loaders import LoaderError, load_dir, load_file
from citerag.retriever import Retriever
from citerag.store import EmbeddingMismatchError, Store
from citerag.validator import Validator

PROVIDERS = {"gemini": "Gemini (free tier)", "anthropic": "Claude"}

# .streamlit/config.toml turns Streamlit's module watcher off, but Streamlit
# restarts it when a browser session reconnects. It then logs a traceback for
# every transformers module that needs torchvision (not installed, not used).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="CiteRAG")


@st.cache_resource(show_spinner="Loading the index and models (first run only)…")
def pipeline():
    store = Store()
    validator = Validator()
    # Load both models now, not during the first question. Assigned, because
    # Streamlit "magic" would draw a bare expression on the page.
    _ = store.embedder.model, validator.nli.model
    return store, Retriever(store), validator


@st.cache_resource
def backend(provider: str):
    return make_backend(provider)


@st.cache_data(max_entries=32, show_spinner=False)
def cited_pages(path: str, mtime: float, first: int, last: int) -> bytes:
    return ui.page_pdf(Path(path), first, last)  # mtime: a changed file isn't served from cache


def flash(kind: str, message: str) -> None:
    """Show a message after the next rerun."""
    st.session_state.flash.append((kind, message))


# -- sidebar -------------------------------------------------------------------------


def sidebar(store: Store) -> dict:
    with st.sidebar:
        st.subheader("Documents")
        docs = store.sources()
        for name, chunks in docs.items():
            left, right = st.columns([6, 1], vertical_alignment="center")
            left.markdown(f"{ui.escape_md(name)}  \n:gray[{chunks} chunk{'s' * (chunks != 1)}]")
            if right.button("", key=f"remove-{name}", icon=":material/close:", type="tertiary",
                            help=f"Remove from the index (the file stays in {config.DATA_DIR.name}/)"):
                store.delete_source(name)
                flash("success", f"Removed {name} from the index.")
                st.rerun()
        if not docs:
            st.caption("No documents indexed yet.")

        files = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
                                 key=f"upload-{st.session_state.upload_key}")
        if files and st.button("Add to index", type="primary", width="stretch"):
            add_files(store, files)
        if st.button(f"Rescan {config.DATA_DIR.name}/ folder", width="stretch",
                     help=f"Index new or changed files in {config.DATA_DIR.name}/ and drop files that are gone"):
            rescan(store)

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
        if st.button("Clear chat", width="stretch", disabled=not st.session_state.turns):
            st.session_state.turns = []
            st.rerun()
    return {"provider": provider, "sources": sources or None, "k": k,
            "validate": validate, "strict": strict and validate}


def add_files(store: Store, files) -> None:
    pages, errors = [], []
    with st.spinner("Reading and indexing…"):
        for f in files:
            try:
                path = ui.save_upload(f.name, f.getvalue(), config.DATA_DIR)
                loaded = load_file(path)
            except (ValueError, LoaderError) as e:
                errors.append(str(e))
                continue
            if not loaded:
                errors.append(f"{path.name}: no text found (a scanned PDF? OCR isn't supported)")
            pages.extend(loaded)
        if pages:
            flash("success", "Indexed. " + ui.report_text(store.index(pages)))
    for e in errors:
        flash("error", e)
    st.session_state.upload_key += 1  # empties the uploader
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


def ask(question: str, settings: dict, retriever: Retriever, validator: Validator) -> Answer | None:
    generator = Generator(retriever, backend(settings["provider"]), validator,
                          validate=settings["validate"], strict=settings["strict"])
    placeholder = st.empty()
    placeholder.caption("Searching…")
    shown: list[str] = []

    def on_text(piece: str) -> None:
        shown.append(piece)
        placeholder.markdown(ui.streaming_markdown("".join(shown)) + " ▌")

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


def show_answer(answer: Answer, turn: int) -> None:
    st.markdown(ui.answer_markdown(answer), unsafe_allow_html=True)
    if line := ui.summary_line(answer):
        st.caption(line)

    hits = {h.chunk_id: h for h in answer.hits}
    for c in answer.citations:
        with st.expander(f"\\[{c.number}\\] {ui.escape_md(c.source)}, {c.pages}"):
            if c.section:
                st.caption(f"Section: {ui.escape_md(c.section)}")
            if c.chunk_id in hits:
                st.markdown(ui.passage_html(hits[c.chunk_id].text, c.start_char, c.end_char),
                            unsafe_allow_html=True)
            pdf = ui.source_pdf(c.source, config.DATA_DIR)
            if pdf is not None and st.toggle(f"Show PDF {c.pages}", key=f"pdf-{turn}-{c.number}"):
                st.pdf(cited_pages(str(pdf), pdf.stat().st_mtime, c.page_start, c.page_end), height=600)

    if answer.validation is not None and (lines := ui.check_lines(answer.validation)):
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
                where = f"{ui.escape_md(h.source)}, {page_label(h)}" + (f" · {ui.escape_md(h.section)}" if h.section else "")
                scores = " · ".join(f"{name} {value:.3f}" for name, value in h.scores.items())
                lines.append(f"{n}. {where} · {'**cited**' if h.chunk_id in cited else 'not cited'}  \n"
                             f"   :gray[{scores}]")
            st.markdown("\n".join(lines))


# -- page ----------------------------------------------------------------------------

st.session_state.setdefault("turns", [])
st.session_state.setdefault("flash", [])
st.session_state.setdefault("upload_key", 0)

st.title("CiteRAG")
st.caption("Ask questions about your documents. Answers cite the passages they come from.")

try:
    store, retriever, validator = pipeline()
except EmbeddingMismatchError as e:
    st.error(str(e))
    st.stop()

settings = sidebar(store)
for kind, message in st.session_state.flash:
    getattr(st, kind)(message)
st.session_state.flash = []

if not store.count():
    st.info("Add a PDF, TXT or MD file in the sidebar to get started.")

for turn, answer in enumerate(st.session_state.turns):
    st.markdown(f"**{ui.escape_md(answer.question)}**")
    show_answer(answer, turn)
    st.divider()

if question := st.chat_input("Ask a question about your documents", disabled=not store.count()):
    st.markdown(f"**{ui.escape_md(question)}**")
    answer = ask(question, settings, retriever, validator)
    if answer is not None:
        st.session_state.turns.append(answer)
        show_answer(answer, len(st.session_state.turns) - 1)
