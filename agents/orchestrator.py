"""
orchestrator.py — SupervisorAgent (the manager) + the public entry points.

This is the travel-project pattern: a supervisor that coordinates specialist
agents, each responsible for one thing:

    SupervisorAgent
        ├─ RetrievalAgent  (plans queries, retrieves evidence)
        ├─ AnalystAgent    (opportunities / risks / trends)
        ├─ AdvisorAgent    (recommendations + CEO briefing)
        └─ QAAgent         (grounded chat answer)

The supervisor routes deterministically by default (guarantees progress, can't
loop forever); set config.LLM_ROUTING=True to let the LLM propose the next step,
validated against the deterministic backbone.

Two modes share one graph:
  briefing : Retrieval -> Analyst -> Advisor   -> analysis.json (the dashboard)
  chat     : Retrieval -> QA                   -> answer (with stateful memory)

Memory:
  graph state (IntelState) is per-run working memory; the SqliteSaver
  checkpointer persists chat turns per thread_id across restarts.

Reasoning runs on a local open model (Qwen2.5 via Ollama). No commercial API.
"""
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

import config as cfg
from agents import _common
from agents import retrieval_agent, analyst_agent, advisor_agent, qa_agent

IntelState = _common.IntelState


# ----------------------------------------------------------------------------- supervisor routing
def _default_next(state: IntelState) -> str:
    """Deterministic routing backbone — guarantees progress, can't loop forever."""
    if not state.get("evidence"):
        return "retrieve"
    if state.get("mode") == "chat":
        return "respond" if not state.get("answer") else "FINISH"
    if not (state.get("opportunities") or state.get("risks")):
        return "analyze"
    if not state.get("recommendations"):
        return "advise"
    return "FINISH"


def supervisor(state: IntelState) -> dict:
    """SupervisorAgent. Deterministic by default; optional LLM routing."""
    fallback = _default_next(state)
    if not getattr(cfg, "LLM_ROUTING", False):
        return {"next": fallback}

    valid = {"retrieve", "analyze", "advise", "respond", "FINISH"}
    summary = {
        "mode": state.get("mode"),
        "have_evidence": bool(state.get("evidence")),
        "have_analysis": bool(state.get("opportunities") or state.get("risks")),
        "have_recommendations": bool(state.get("recommendations")),
        "have_answer": bool(state.get("answer")),
        "valid_steps": sorted(valid),
    }
    raw = _common.invoke([
        {"role": "system", "content": "You route a strategic-intelligence workflow. "
         "Reply ONLY as JSON: {\"next\": \"<one valid step>\"}."},
        {"role": "user", "content": json.dumps(summary)},
    ])
    decision = _common.parse_json(raw, {}).get("next", fallback)
    return {"next": decision if decision in valid else fallback}


# ----------------------------------------------------------------------------- graph + memory
def get_checkpointer():
    """Persistent memory across restarts; falls back to in-memory if sqlite saver absent."""
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(cfg.CHECKPOINT_DB, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver
        print(f"  (sqlite checkpointer unavailable: {e}; using in-memory)")
        return MemorySaver()


def build_graph(checkpointer=None):
    """Wire the supervisor to the four specialist agents (sub_agents equivalent)."""
    b = StateGraph(IntelState)
    b.add_node("supervisor", supervisor)
    b.add_node("retrieve", retrieval_agent.run)   # RetrievalAgent
    b.add_node("analyze", analyst_agent.run)      # AnalystAgent
    b.add_node("advise", advisor_agent.run)       # AdvisorAgent
    b.add_node("respond", qa_agent.run)           # QAAgent

    b.add_edge(START, "supervisor")
    b.add_conditional_edges(
        "supervisor", lambda s: s["next"],
        {"retrieve": "retrieve", "analyze": "analyze", "advise": "advise",
         "respond": "respond", "FINISH": END},
    )
    for worker in ("retrieve", "analyze", "advise", "respond"):
        b.add_edge(worker, "supervisor")   # specialists report back to the supervisor

    return b.compile(checkpointer=checkpointer or get_checkpointer())


# Module-level compiled graph (LangGraph Studio entry point).
graph = build_graph()

_CHAT_GRAPH = None   # reused across chat turns so the checkpointer connection persists


# ----------------------------------------------------------------------------- public entry points
def run_briefing(use_llm=True, temperature=None, k=None) -> dict:
    """Full board briefing through the supervisor + specialist team.

    Returns the analysis.json-shaped dict the dashboard reads.
    """
    if not use_llm:
        return _fallback_briefing()
    if k is not None:
        cfg.RETRIEVAL_K = int(k)

    g = build_graph()
    run_cfg = {"configurable": {"thread_id": f"briefing::{uuid.uuid4()}"},
               "recursion_limit": getattr(cfg, "RECURSION_LIMIT", 25)}
    final = g.invoke({
        "company": cfg.COMPANY, "industry": cfg.INDUSTRY,
        "question": cfg.BRIEFING_QUESTION, "mode": "briefing", "messages": [],
        "temperature": cfg.TEMPERATURE if temperature is None else temperature,
    }, run_cfg)

    idx = final.get("index_map", {})
    result = {
        "opportunities": _common.attach_evidence(final.get("opportunities", []), idx),
        "risks": _common.attach_evidence(final.get("risks", []), idx),
        "trends": _common.attach_evidence(final.get("trends", []), idx),
        "recommendations": _common.attach_evidence(final.get("recommendations", []), idx),
        "ceo_briefing": final.get("ceo_briefing", {}),
        "company": cfg.COMPANY, "industry": cfg.INDUSTRY,
        "_engine": "specialist_team",
    }
    # if the team produced nothing usable, fall back so the dashboard still renders
    if not (result["opportunities"] or result["risks"] or result["recommendations"]):
        fb = _fallback_briefing()
        fb["_engine"] = "specialist_team_fallback"
        return fb
    return result


def chat(question: str, thread_id="ceo-chat", temperature=None, use_llm=True) -> str:
    """Interactive Q&A with stateful memory. Reuse a thread_id to keep context.

    Routes Retrieval -> QA through the same graph; the checkpointer persists
    conversation history per thread_id across turns and restarts.
    """
    global _CHAT_GRAPH
    if _CHAT_GRAPH is None:
        _CHAT_GRAPH = build_graph()
    run_cfg = {"configurable": {"thread_id": thread_id},
               "recursion_limit": getattr(cfg, "RECURSION_LIMIT", 25)}
    final = _CHAT_GRAPH.invoke({
        "company": cfg.COMPANY, "industry": cfg.INDUSTRY,
        "question": question, "mode": "chat",
        "messages": [HumanMessage(content=question)],
        "evidence": [], "answer": "",
        "temperature": cfg.TEMPERATURE if temperature is None else temperature,
    }, run_cfg)
    return final.get("answer", "")


def _fallback_briefing() -> dict:
    """Deterministic placeholder so the dashboard renders without a running LLM."""
    return {
        "opportunities": [{"title": "Expand AI-embedded product portfolio", "impact": "High",
                           "confidence": 0.6, "rationale": "Placeholder — start Ollama for real reasoning.",
                           "evidence": []}],
        "risks": [{"title": "Intensifying competition", "category": "Competitive", "severity": "High",
                   "confidence": 0.6, "rationale": "Placeholder.", "evidence": []}],
        "trends": [{"title": "Enterprise shift to agentic AI", "evidence": []}],
        "recommendations": [{"action": "Accelerate AI go-to-market", "priority": "High",
                             "expected_impact": "Revenue growth and differentiation",
                             "risk_level": "Medium", "evidence": []}],
        "ceo_briefing": {"what_happened": "LLM unavailable — deterministic placeholder.",
                         "why_it_matters": "Run Ollama (qwen2.5:7b) to generate real reasoning.",
                         "what_to_do_next": "Start Ollama, then: python pipeline.py --analyze-only"},
        "company": cfg.COMPANY, "industry": cfg.INDUSTRY, "_fallback": True,
    }


if __name__ == "__main__":
    import sys
    out = run_briefing(use_llm="--no-llm" not in sys.argv)
    print(json.dumps(out.get("ceo_briefing", {}), indent=2, ensure_ascii=False))
