"""
sentiment.py — sentiment tool reading the pipeline's sentiment.json.

The heavy lifting (VADER/FinBERT scoring, per-source aggregation) already happens
in engine/sentiment.py during the pipeline. This tool just surfaces that result
to the agent so it can ground its reasoning in measured corpus mood instead of
guessing.
"""
import json
import os

from langchain_core.tools import tool

import config as cfg


@tool
def get_sentiment(_: str = "") -> str:
    """Get the corpus-wide sentiment summary, overall and per source.

    Returns the overall compound sentiment and a per-source breakdown
    (positive/neutral/negative counts and average compound), as computed by the
    pipeline's sentiment stage.
    """
    path = os.path.join(cfg.DATA_DIR, "sentiment.json")
    if not os.path.exists(path):
        return "No sentiment data yet. Run the pipeline's sentiment stage first."
    data = json.load(open(path, encoding="utf-8"))
    overall = data.get("overall_compound", 0.0)
    engine = data.get("engine", "?")
    lines = [f"Overall sentiment: {overall:+.3f} (engine={engine})", "By source:"]
    for src, b in sorted(data.get("by_source", {}).items()):
        lines.append(
            f"  {src}: avg={b.get('avg_compound', 0):+.3f} "
            f"(+{b.get('positive', 0)}/={b.get('neutral', 0)}/-{b.get('negative', 0)}, "
            f"n={b.get('total', 0)})"
        )
    return "\n".join(lines)
