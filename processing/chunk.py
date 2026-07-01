"""
Document chunking for retrieval.

The clean step collapses each document to a single whitespace-normalized line,
so we chunk on sentence boundaries (recursive: sentences -> words -> characters)
with a character overlap to preserve context across chunk borders. This is the
dependency-free equivalent of LangChain's RecursiveCharacterTextSplitter for
already-normalized text.

Why chunk: embedding a whole 1500-word article into one vector blurs many topics
into a single point, so retrieval is coarse and the bge model truncates anything
past its context window. Splitting into ~800-char passages gives the RAG agent
focused, on-topic evidence and avoids silent truncation.

Each chunk carries its parent document's metadata (source/title/url/published)
plus parent_id and chunk_index, so evidence resolution still points back to the
original source.
"""
import sys, os, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _split_sentences(text):
    # split after . ! ? followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _hard_split(s, size, overlap):
    # fixed character window with overlap, for sentences longer than `size`
    out, start, step = [], 0, max(1, size - overlap)
    while start < len(s):
        out.append(s[start:start + size])
        start += step
    return out


def split_text(text, size=None, overlap=None):
    """Split one document's text into overlapping, sentence-aware chunks."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, cur = [], ""
    for sent in _split_sentences(text):
        if len(sent) > size:                      # sentence alone exceeds target
            if cur:
                chunks.append(cur.strip()); cur = ""
            chunks.extend(_hard_split(sent, size, overlap))
            continue
        if len(cur) + len(sent) + 1 <= size:      # fits in current chunk
            cur = f"{cur} {sent}".strip()
        else:                                      # start a new chunk with overlap tail
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail} {sent}".strip()
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def chunk_docs(docs, size=None, overlap=None):
    """Expand a list of documents into chunk records, preserving source metadata."""
    out = []
    for d in docs:
        pieces = split_text(d.get("text", ""), size, overlap)
        for i, piece in enumerate(pieces):
            out.append({
                "id": f"{d['id']}::c{i}",
                "parent_id": d["id"],
                "chunk_index": i,
                "source": d.get("source", ""),
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "published": d.get("published", ""),
                "text": piece,
            })
    return out


if __name__ == "__main__":
    sample = (
        "SAP reported strong cloud revenue growth this quarter. " * 3
        + "Analysts remain cautious about competition from Oracle and Salesforce. " * 4
    )
    pieces = split_text(sample, size=200, overlap=40)
    print(f"{len(sample)} chars -> {len(pieces)} chunks")
    for i, p in enumerate(pieces):
        print(f"  [{i}] ({len(p)}c) {p[:70]}...")
