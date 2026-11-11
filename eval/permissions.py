"""Permission leak test (Phase 13): restricted users must never get a passage they may not read.

Every question in eval/questions.json is searched as several users, each
allowed only some of the evaluation documents, through the app's real search
(dense + keyword + reranker) wrapped in access.ScopedRetriever. Any passage
from a document outside the user's set is a leak; the expected count is 0.
Also reported: how many questions about documents a user can't read found
nothing at all (the app answers "not in the documents" without calling the LLM).

Local only: no LLM, no API key.

Run:  python -m eval.permissions
"""

from __future__ import annotations

import logging

from citerag import config
from citerag.access import ScopedRetriever
from citerag.retriever import Reranker, Retriever
from citerag.store import Embedder
from eval.metrics import load_questions
from eval.retrieval import QUESTIONS, load_documents, open_index

PROFILES = {  # user -> the evaluation documents they may read
    "papers only": ["NIPS-2017-attention-is-all-you-need-Paper.pdf",
                    "Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.pdf"],
    "company files only": ["leave-and-travel-policy.docx", "q3-business-review.pptx", "pokhara-office-lease.eml"],
    "syllabus only": ["bsccsit-syllabus-webtechnology.pdf"],
    "nothing": [],
}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    documents, questions = load_questions(QUESTIONS)
    pages, _ = load_documents(documents)
    store = open_index(config.CHUNK_SIZE, pages, Embedder())
    retriever = Retriever(store, reranker=Reranker(), use_reranker=True)
    for name, allowed in PROFILES.items():
        scoped = ScopedRetriever(retriever, allowed)
        leaks, searched, off_limits, empty = 0, 0, 0, 0
        for q in questions:
            hits = scoped.retrieve(q.question, k=config.TOP_K)
            searched += 1
            leaks += sum(h.source not in allowed for h in hits)
            if q.sources and not q.sources & set(allowed):  # a question about documents this user can't read
                off_limits += 1
                empty += not hits
        print(f"{name:<20} {searched} questions: {leaks} passages from other documents | "
              f"questions about documents they can't read: {off_limits}, of which found nothing: {empty}")


if __name__ == "__main__":
    main()
