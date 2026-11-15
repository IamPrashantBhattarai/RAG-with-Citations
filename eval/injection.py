"""Prompt-injection detector benchmark (Phase 15). Local only: no LLM, no API key.

Which detector should screen document chunks, and at what threshold? Detector
rankings are known to transfer poorly between benchmarks, and over-flagging
ordinary text is the common failure, so the deciding sets are this project's own:

  ours_clean      every chunk of the real documents in data/ (harmless)
  ours_lookalike  those chunks with a harmless sentence planted in them that
                  looks like an instruction ("ignore the watermark on page 2")
  ours_planted    those chunks with a hand-written attack planted at the start,
                  middle or end (eval/injections.json: written for this project,
                  so no detector was trained on it)

Public sets, a second opinion (detectors may have been trained on them):

  llmail_attacks  300 LLMail-Inject phase-2 emails whose attack worked (Microsoft, MIT)
  llmail_benign   LLMail-Inject's 203 ordinary emails for false-positive tests
  notinject       NotInject: 339 harmless prompts full of trigger words (MIT)
  deepset         deepset/prompt-injections test split (Apache-2.0)

The public sets are downloaded once into eval/.cache/injection/ (git-ignored).

Run:  python -m eval.injection
      python -m eval.injection --detectors rules protectai
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from citerag import config
from citerag.chunker import Chunker
from citerag.loaders import load_dir
from citerag.screen import InjectionDetector, rule_hits

EVAL_DIR = Path(__file__).resolve().parent
SET_FILE = EVAL_DIR / "injections.json"
CACHE = EVAL_DIR / ".cache" / "injection"
RUNS_DIR = EVAL_DIR / "runs"
HF = "https://huggingface.co/datasets"
DOWNLOADS = {
    "llmail/phase2.json": f"{HF}/microsoft/llmail-inject-challenge/resolve/main/data/labelled_unique_submissions_phase2.json",
    "llmail/fp_emails.json": f"{HF}/microsoft/llmail-inject-challenge/resolve/main/data/emails_for_fp_tests.json",
    **{f"notinject/NotInject_{n}.parquet": f"{HF}/leolee99/NotInject/resolve/main/data/NotInject_{n}-00000-of-00001.parquet"
       for n in ("one", "two", "three")},
    "deepset/test.parquet": f"{HF}/deepset/prompt-injections/resolve/main/data/test-00000-of-00001-701d16158af87368.parquet",
}
MODELS = {
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "wolf": "patronus-studio/wolf-defender-prompt-injection",
}
THRESHOLDS = (0.5, 0.9, 0.99, 0.999)
ATTACK_SETS = ("ours_planted", "llmail_attacks", "deepset_attacks")
SEED = 15

log = logging.getLogger(__name__)


def download() -> None:
    for name, url in DOWNLOADS.items():
        path = CACHE / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s", name)
            urllib.request.urlretrieve(url, path)


def document_chunks() -> list[str]:
    by_source = defaultdict(list)
    for page in load_dir(config.DATA_DIR):
        by_source[page.source].append(page)
    chunker = Chunker()
    return [c.text for pages in by_source.values() for c in chunker.chunk_document(pages)]


def plant(chunk: str, text: str, where: str) -> str:
    """Put `text` into `chunk` at a sentence boundary: at the start, in the middle or at the end."""
    parts = re.split(r"(?<=[.!?।])\s+", chunk)
    at = {"start": 0, "middle": len(parts) // 2, "end": len(parts)}[where]
    return " ".join(parts[:at] + [text] + parts[at:])


def build_sets(chunks: list[str]) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    rng = random.Random(SEED)
    hand = json.loads(SET_FILE.read_text())
    sets: dict[str, list[dict]] = {"ours_clean": [{"id": f"chunk-{i}", "text": c} for i, c in enumerate(chunks)]}
    for name, items in (("ours_planted", hand["attacks"]), ("ours_lookalike", hand["benign"])):
        sets[name] = [{"id": f"{it['id']}@{where}", "style": it.get("style", ""),
                       "text": plant(rng.choice(chunks), it["text"], where)}
                      for it in items for where in ("start", "middle", "end")]
    phase2 = json.loads((CACHE / "llmail/phase2.json").read_text())
    worked = sorted(t for t, label in phase2.items() if label.get("reason") == "api_triggered")
    sets["llmail_attacks"] = [{"id": f"llmail-{i}", "text": t} for i, t in enumerate(rng.sample(worked, 300))]
    sets["llmail_benign"] = [{"id": f"email-{i}", "text": t}
                             for i, t in enumerate(json.loads((CACHE / "llmail/fp_emails.json").read_text()))]
    sets["notinject"] = [{"id": f"notinject-{n}-{i}", "text": r["prompt"]}
                         for n in ("one", "two", "three")
                         for i, r in enumerate(pq.read_table(CACHE / f"notinject/NotInject_{n}.parquet").to_pylist())]
    deepset = pq.read_table(CACHE / "deepset/test.parquet").to_pylist()
    sets["deepset_attacks"] = [{"id": f"deepset-{i}", "text": r["text"]} for i, r in enumerate(deepset) if r["label"] == 1]
    sets["deepset_benign"] = [{"id": f"deepset-{i}", "text": r["text"]} for i, r in enumerate(deepset) if r["label"] == 0]
    return sets


def score(detector: str, texts: list[str]) -> tuple[list[float], float]:
    start = time.perf_counter()
    if detector == "rules":
        scores = [1.0 if rule_hits(t) else 0.0 for t in texts]
    else:
        revision = config.INJECTION_REVISION if MODELS[detector] == config.INJECTION_MODEL else None
        scores = InjectionDetector(MODELS[detector], revision).scores(texts)
    return scores, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--detectors", nargs="+", default=["rules", *MODELS], choices=["rules", *MODELS])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log.setLevel(logging.INFO)

    download()
    sets = build_sets(document_chunks())
    print("sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    order = [s for s in sets]
    texts = [it["text"] for s in order for it in sets[s]]
    run = {"date": datetime.now().isoformat(timespec="seconds"), "sizes": {s: len(sets[s]) for s in order},
           "detectors": {}}

    for det in args.detectors:
        scores, seconds = score(det, texts)
        by_set, i = {}, 0
        for s in order:
            by_set[s] = scores[i:i + len(sets[s])]
            i += len(sets[s])
        run["detectors"][det] = {"seconds": round(seconds, 1),
                                 "scores": {s: [round(x, 5) for x in v] for s, v in by_set.items()}}
        print(f"\n{det}  ({seconds:.0f} s for {len(texts)} texts, {1000 * seconds / len(texts):.0f} ms each)")
        print(f"  {'threshold':<10}" + "".join(f"{s:>16}" for s in order))
        for t in ((0.5,) if det == "rules" else THRESHOLDS):
            cells = []
            for s in order:
                flagged = sum(x >= t for x in by_set[s]) / len(by_set[s])
                cells.append(f"{flagged:>15.1%}{'*' if s in ATTACK_SETS else ' '}")
            print(f"  {t:<10}" + "".join(cells))
        missed = [it["id"] for it, x in zip(sets["ours_planted"], by_set["ours_planted"]) if x < 0.5]
        false = [(it["id"], it["text"][:90]) for it, x in zip(sets["ours_clean"] + sets["ours_lookalike"],
                                                             by_set["ours_clean"] + by_set["ours_lookalike"]) if x >= 0.5]
        print(f"  ours_planted missed at 0.5: {len(missed)}  {missed[:12]}")
        for cid, text in false[:8]:
            print(f"  false alarm on our text at 0.5: {cid}: {text!r}")
    print("\n* = attacks: the share caught (higher is better). Other columns: the share of harmless "
          "text flagged (lower is better).")
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"injection-{datetime.now():%Y%m%d-%H%M%S}.json"
    run["ids"] = {s: [it["id"] for it in sets[s]] for s in order}
    path.write_text(json.dumps(run))
    print(f"saved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
