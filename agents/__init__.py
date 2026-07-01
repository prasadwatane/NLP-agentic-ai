"""
agents — the specialist agent team (travel-project style).

Each specialist is responsible for one thing, coordinated by a supervisor:

  orchestrator.py     SupervisorAgent — routes to the specialists, owns the graph
  retrieval_agent.py  RetrievalAgent  — plans queries (LLM) + retrieves evidence
  analyst_agent.py    AnalystAgent    — opportunities / risks / trends (LLM)
  advisor_agent.py    AdvisorAgent    — recommendations + CEO briefing (LLM)
  qa_agent.py         QAAgent         — grounded chat answer (LLM)
  _common.py          shared state, LLM invoke, JSON parse, evidence helpers

Public API (used by pipeline.py and streamlit_app.py):
  run_briefing(use_llm, temperature, k) -> analysis.json-shaped dict
  chat(question, thread_id, temperature) -> str
  graph                                  -> compiled supervisor graph (Studio)
"""
from agents.orchestrator import build_graph, chat, graph, run_briefing

# Specialist agents are importable by name for the oral exam / live coding.
from agents import retrieval_agent, analyst_agent, advisor_agent, qa_agent

__all__ = [
    "run_briefing", "chat", "graph", "build_graph",
    "retrieval_agent", "analyst_agent", "advisor_agent", "qa_agent",
]
