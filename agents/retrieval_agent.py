"""
retrieval_agent.py — RetrievalAgent (specialist).

Responsibility:
    Gather the evidence the rest of the team reasons over. Unlike a fixed
    retrieval step, this agent makes its OWN LLM call to PLAN what to search for
    (given the company + mode), then runs those searches against the ChromaDB
    knowledge repository via the search_corpus tool.

Tools owned:
    search_corpus(query, k)   — hybrid retrieval over the indexed corpus
    gather_evidence(company)  — themed fallback pack if planning yields nothing

Analogy: like the travel project's WeatherAgent owning get_weather — this agent
owns retrieval and decides how to use it.
"""
import json

import config as cfg
from agents import _common
from tools import search_corpus
from retrieval import gather_evidence, hybrid_search

PLANNER_PROMPT = """You plan an evidence search about {company} ({industry}).
Propose 4 focused search queries that together cover: growth opportunities,
competitive and regulatory risks, and technology/industry trends.

Return ONLY this JSON object:
{{"queries": ["query 1", "query 2", "query 3", "query 4"]}}
Each query is a short phrase of keywords (not a sentence). JSON only."""


def plan_queries(company: str, industry: str, temperature=None) -> list:
    """LLM call: decide which searches to run. Falls back to themed defaults."""
    msgs = [
        {"role": "system", "content": _common.SYSTEM.format(company=company, industry=industry)},
        {"role": "user", "content": PLANNER_PROMPT.format(company=company, industry=industry)},
    ]
    data = _common.parse_json(_common.invoke(msgs, temperature=temperature), {})
    queries = data.get("queries") or []
    # keep only sensible strings; fall back if the model gave us nothing usable
    queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 3][:6]
    return queries


def run(state: _common.IntelState) -> dict:
    """Plan (LLM) -> retrieve (tool). Briefing plans a themed sweep; chat retrieves per question."""
    company = state["company"]
    temperature = state.get("temperature", cfg.TEMPERATURE)

    if state.get("mode") == "chat":
        # chat: retrieve directly for the question (no planning needed)
        evidence = {"query": hybrid_search(state["question"], k=cfg.RETRIEVAL_K)}
        plan = [state["question"]]
    else:
        # briefing: the agent decides what to search for, then retrieves each
        plan = plan_queries(company, state["industry"], temperature)
        if plan:
            evidence = {}
            for i, q in enumerate(plan, 1):
                hits = search_corpus.invoke({"query": q, "k": cfg.RETRIEVAL_K})
                # search_corpus returns a formatted string; re-run hybrid_search for structured hits
                evidence[f"q{i}:{q[:40]}"] = hybrid_search(q, k=cfg.RETRIEVAL_K)
        else:
            # planning produced nothing usable -> themed fallback pack
            evidence = gather_evidence(company)

    block, index_map = _common.build_evidence_block(evidence)
    flat = [{"tag": t, **m} for t, m in index_map.items()]
    return {"evidence": flat, "evidence_block": block, "index_map": index_map,
            "search_plan": plan}
