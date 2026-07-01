"""
Embedding + indexing. Encodes the clean corpus with BAAI/bge-small-en-v1.5
and upserts into a persistent ChromaDB collection for semantic retrieval.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def get_collection(reset=False):
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index(reset=True):
    corpus_path = os.path.join(config.DATA_DIR, "clean_corpus.json")
    if not os.path.exists(corpus_path):
        sys.exit("No clean_corpus.json. Run processing/clean.py first.")
    docs = json.load(open(corpus_path, encoding="utf-8"))

    # Split documents into overlapping, sentence-aware chunks for finer retrieval.
    from processing.chunk import chunk_docs
    chunks = chunk_docs(docs)
    print(f"  indexing {len(docs)} docs -> {len(chunks)} chunks "
          f"(size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP})")

    model = get_embedder()
    col = get_collection(reset=reset)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    ).tolist()

    # Chroma metadata values must be scalars. parent_id/chunk_index let a chunk
    # be traced back to its source document during evidence resolution.
    metas = []
    for c in chunks:
        metas.append({
            "source": str(c.get("source", "")),
            "title": str(c.get("title", ""))[:300],
            "url": str(c.get("url", "")),
            "published": str(c.get("published", "")),
            "parent_id": str(c.get("parent_id", "")),
            "chunk_index": int(c.get("chunk_index", 0)),
        })

    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=metas,
    )
    print(f"  collection '{config.COLLECTION_NAME}' now holds {col.count()} chunks "
          f"from {len(docs)} documents")


if __name__ == "__main__":
    index()
