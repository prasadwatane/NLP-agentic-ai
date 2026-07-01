"""
Retriever. Provides semantic search over ChromaDB plus an optional hybrid
(semantic + keyword) mode. Modular by design so the oral-exam 'add hybrid
search' / 'add a filter' asks are a few-line change.
"""
import sys, os, re
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


def semantic_search(query, k=8, source_filter=None):
    model = _embedder()
    q = model.encode([query], normalize_embeddings=True).tolist()
    kwargs = {"query_embeddings": q, "n_results": k}
    if source_filter:
        kwargs["where"] = {"source": source_filter}
    res = _collection().query(**kwargs)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "text": doc,
            "meta": meta,
            "score": 1 - dist,   # cosine distance -> similarity
        })
    return hits


def _keyword_score(query, text):
    terms = set(re.findall(r"\w+", query.lower()))
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    overlap = sum(1 for w in words if w in terms)
    return overlap / len(words)


def hybrid_search(query, k=8, alpha=0.7, source_filter=None):
    """alpha weights semantic vs keyword. 1.0 = pure semantic."""
    # Over-fetch then re-rank with a keyword signal.
    pool = semantic_search(query, k=k * 3, source_filter=source_filter)
    for h in pool:
        kw = _keyword_score(query, h["text"])
        h["score"] = alpha * h["score"] + (1 - alpha) * kw
    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool[:k]


def gather_evidence(company, k_per_theme=6):
    """Pull a themed evidence pack the CEO agent can reason over."""
    themes = {
        "opportunities": f"{company} growth opportunities new markets AI products partnerships",
        "risks": f"{company} risks competition regulation lawsuit decline threat",
        "competitors": f"{company} competitors {' '.join(config.COMPETITORS)}",
        "trends": f"{company} technology trends industry shift customer demand",
    }
    evidence = {}
    for theme, q in themes.items():
        evidence[theme] = hybrid_search(q, k=k_per_theme)
    return evidence


if __name__ == "__main__":
    for h in hybrid_search(f"{config.COMPANY} biggest risks", k=5):
        print(f"[{h['score']:.3f}] ({h['meta']['source']}) {h['meta']['title'][:80]}")
