"""
test_context.py — sweep retrieval breadth k ("context size") and compare.

Same lab methodology, different knob: hold the query fixed, vary how much context
the agent retrieves per search (k), run, and compare. Writes
reports/context_sweep.txt for the oral exam.

The retrieval layer (engine.retriever.hybrid_search) needs only the ChromaDB
index, not the LLM — so the core monotonicity test runs without Ollama. A second
test exercises the full agent at different k when Ollama is available.

Hypothesis under test:
  * more context (larger k) returns at least as many evidence snippets as smaller
    k (monotonic non-decreasing), up to corpus size.
"""
import os

import pytest

import config as cfg
from tools import search_corpus

K_VALUES = [2, 4, 8, 12]
FIXED_QUERY = f"{cfg.COMPANY} competitive risks and growth opportunities"

_REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _snippet_count(tool_output: str) -> int:
    # search_corpus formats each hit as a line starting with "[n] source=..."
    return sum(1 for line in tool_output.splitlines() if line.strip().startswith("[") and "source=" in line)


@pytest.mark.parametrize("k", K_VALUES)
def test_search_returns_results_at_each_k(require_index, k):
    out = search_corpus.invoke({"query": FIXED_QUERY, "k": k})
    assert isinstance(out, str) and out.strip()
    assert _snippet_count(out) <= k  # never more than requested


def test_context_monotonic_and_write_comparison(require_index):
    """Evidence count is non-decreasing in k; save a side-by-side artifact."""
    os.makedirs(_REPORTS, exist_ok=True)
    counts, rows = {}, []
    for k in K_VALUES:
        out = search_corpus.invoke({"query": FIXED_QUERY, "k": k})
        counts[k] = _snippet_count(out)
        rows.append(f"\n=== k={k} (retrieved {counts[k]} snippets) ===\n{out}\n")

    out_path = os.path.join(_REPORTS, "context_sweep.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"CONTEXT (k) SWEEP · {cfg.COMPANY}\nQuery: {FIXED_QUERY}\n")
        f.write("".join(rows))
    assert os.path.exists(out_path)

    # Monotonic non-decreasing up to available corpus.
    ordered = [counts[k] for k in K_VALUES]
    assert all(b >= a for a, b in zip(ordered, ordered[1:])), f"non-monotonic: {counts}"


def test_agent_brief_varies_with_context(require_ollama, require_index):
    """Full briefing runs at small vs large k and produces valid analysis both times."""
    small = agent_brief(k=2)
    large = agent_brief(k=10)
    for res in (small, large):
        assert res.get("opportunities") is not None
        assert "ceo_briefing" in res


def agent_brief(k):
    import agents as agent
    return agent.run_briefing(use_llm=True, temperature=0.2, k=k)
