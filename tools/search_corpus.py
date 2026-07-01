"""
search_corpus.py — the retrieval tool.

@tool-decorated so the agent can call it. Wraps the project's ChromaDB hybrid
search and returns human-readable, source-tagged snippets (not raw dicts) so the
model can copy the title/source/url into its final StrategicAnalysis evidence.
"""
from langchain_core.tools import tool

import config as cfg
from retrieval import hybrid_search


@tool
def search_corpus(query: str, k: int = None) -> str:
    """Search the knowledge base for evidence about the company.

    Use this several times with different queries to investigate distinct angles
    (growth opportunities, competitive risks, regulation, technology trends).

    Args:
        query: A focused search query, e.g. "SAP cloud margin pressure competition".
        k: How many snippets to return (default uses the configured retrieval breadth).

    Returns:
        Numbered, source-tagged evidence snippets. Copy the relevant title/source/url
        into your final analysis as evidence.
    """
    k = cfg.RETRIEVAL_K if k is None else int(k)
    hits = hybrid_search(query, k=k)
    if not hits:
        return "No evidence found for that query."
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        snippet = (h.get("text", "") or "")[:300].replace("\n", " ")
        lines.append(
            f"[{i}] source={meta.get('source', '?')} | "
            f"title={meta.get('title', '')[:100]} | "
            f"url={meta.get('url', '')}\n    {snippet}"
        )
    return "\n".join(lines)
