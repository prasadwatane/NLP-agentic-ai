"""
tools — the agent's tool layer, one tool per file.

  search_corpus.py  -> search_corpus   (ChromaDB hybrid retrieval)
  list_sources.py   -> list_sources    (per-source document coverage)
  get_sentiment.py  -> get_sentiment   (corpus sentiment from sentiment.json)

Each tool is a thin wrapper over the project's existing data layer, so the agent
reasons over the SAME index and sentiment the dashboard shows.
"""
from tools.search_corpus import search_corpus
from tools.list_sources import list_sources
from tools.get_sentiment import get_sentiment

# Data-gathering tools the agent loops over. The terminal StrategicAnalysis tool
# is bound separately in the agent (it's a Pydantic schema, not a function).
DATA_TOOLS = [search_corpus, get_sentiment, list_sources]

__all__ = ["search_corpus", "get_sentiment", "list_sources", "DATA_TOOLS"]
