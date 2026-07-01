"""
Cleaning + deduplication. Merges all collector outputs, removes exact and
near-duplicate documents, drops empties, and writes a single clean corpus.
"""
import sys, os, json, glob, re, hashlib
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def normalize(text):
    text = re.sub(r"http\S+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _near_dup(a, b, threshold=0.88):
    # Compare full titles; require high similarity so only genuine
    # republications/syndications are dropped, not distinct articles
    # that happen to share a few words.
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


def load_all():
    # Only collector outputs are documents; skip pipeline sidecar files and
    # guard against any non-list / non-document JSON landing in data/.
    skip = {"clean_corpus.json", "analysis.json", "sentiment.json"}
    docs = []
    for f in glob.glob(os.path.join(config.DATA_DIR, "*.json")):
        if os.path.basename(f) in skip:
            continue
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not load {f}: {e}")
            continue
        if not isinstance(data, list):
            print(f"  ! skipping {os.path.basename(f)} (not a document list)")
            continue
        docs.extend(d for d in data if isinstance(d, dict) and "text" in d)
    return docs


def clean():
    raw = load_all()
    print(f"  loaded {len(raw)} raw docs")

    # 1. normalize + drop empties / too-short
    cleaned = []
    for d in raw:
        d["text"] = normalize(d.get("text", ""))
        d["title"] = normalize(d.get("title", ""))
        if len(d["text"]) < 30:
            continue
        cleaned.append(d)

    # 2. exact dedup by id
    by_id = {}
    for d in cleaned:
        by_id[d["id"]] = d
    cleaned = list(by_id.values())
    print(f"  after exact dedup: {len(cleaned)}")

    # 3. near-duplicate dedup (title-based, keeps first seen)
    kept = []
    seen_titles = []
    for d in sorted(cleaned, key=lambda x: len(x["text"]), reverse=True):
        if any(_near_dup(d["title"], t) for t in seen_titles):
            continue
        seen_titles.append(d["title"])
        kept.append(d)
    print(f"  after near-dup dedup: {len(kept)}")

    out = os.path.join(config.DATA_DIR, "clean_corpus.json")
    json.dump(kept, open(out, "w"), indent=2, ensure_ascii=False)

    # rubric check
    sources = {d["source"] for d in kept}
    print(f"\n  === CORPUS SUMMARY ===")
    print(f"  documents : {len(kept)}  (min required {config.MIN_DOCUMENTS})")
    print(f"  sources   : {len(sources)} {sorted(sources)}  (min required {config.MIN_SOURCES})")
    if len(kept) < config.MIN_DOCUMENTS:
        print(f"  WARNING: below 100 docs. Re-run collectors or add feeds.")
    return kept


if __name__ == "__main__":
    clean()
