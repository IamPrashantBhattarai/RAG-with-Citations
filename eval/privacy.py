"""Private data in the documents (Phase 15): what the masking would hide, file by file.

Runs citerag.privacy over every document in data/ and lists each finding with
the words before it, masked, so false alarms (a year range taken for a phone
number) and misses can be checked by eye. Local only; nothing is sent anywhere.

Run:  python -m eval.privacy
"""

from __future__ import annotations

import logging
from collections import Counter

from citerag import config
from citerag.loaders import load_dir
from citerag.privacy import find_private, mask_value


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    pages = load_dir(config.DATA_DIR)
    counts: Counter = Counter()
    for p in pages:
        for f in find_private(p.text):
            counts[(p.source, f.kind)] += 1
            before = " ".join(p.text[max(0, f.start - 50):f.start].split())
            print(f"{p.source}  {f.label()}:  …{before} {mask_value(f.kind, p.text[f.start:f.end])}")
    sources = sorted({p.source for p in pages})
    print(f"\n{len(sources)} documents, {len(pages)} pages")
    for source in sources:
        found = {kind: n for (s, kind), n in counts.items() if s == source}
        print(f"  {source}: {', '.join(f'{n} {k}' for k, n in found.items()) or 'nothing'}")


if __name__ == "__main__":
    main()
