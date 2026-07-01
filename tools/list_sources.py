"""
list_sources.py — coverage tool.

Reports how many documents exist per source channel, so the agent can gauge how
well-evidenced the corpus is before concluding.
"""
import glob
import json
import os

from langchain_core.tools import tool

import config as cfg


@tool
def list_sources(_: str = "") -> str:
    """List how many documents exist per source channel, to gauge coverage.

    Returns a count per source (news, company_pr, hackernews, github, etc.),
    read from the collected data files.
    """
    counts = {}
    skip = {"clean_corpus.json", "analysis.json", "sentiment.json"}
    for path in sorted(glob.glob(os.path.join(cfg.DATA_DIR, "*.json"))):
        name = os.path.basename(path)
        if name in skip:
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    src = d.get("source", name.replace(".json", ""))
                    counts[src] = counts.get(src, 0) + 1
    if not counts:
        return "No collected documents found. Run the pipeline first."
    total = sum(counts.values())
    rows = "\n".join(f"  {src}: {n}" for src, n in sorted(counts.items()))
    return f"Document coverage ({total} total across {len(counts)} sources):\n{rows}"
