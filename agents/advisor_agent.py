"""
advisor_agent.py — AdvisorAgent (specialist).

Responsibility:
    Take the analyst's signals and produce the executive deliverable: ruthlessly
    prioritized, evidence-backed recommendations plus the CEO briefing
    (what happened / why it matters / what to do next). Its own LLM call.

Input  : state["opportunities"], state["risks"], state["trends"]
Output : state["recommendations"], state["ceo_briefing"]
"""
import json

import config as cfg
from agents import _common
from langchain_core.messages import AIMessage

ADVISOR_PROMPT = """You have this analysis of {company}:
{analysis}

Grounded in the same evidence tags, produce prioritized recommendations and a CEO briefing.
Return ONLY this JSON object:
{{
  "recommendations": [{{"action": "specific strategic action", "priority": "High|Medium|Low",
                        "evidence_ids": ["E1"], "expected_impact": "revenue/market/customer impact",
                        "risk_level": "High|Medium|Low"}}],
  "ceo_briefing": {{"what_happened": "2-3 sentences", "why_it_matters": "2-3 sentences",
                    "what_to_do_next": "2-3 sentences, the single most important move"}}
}}
Rules: 4-6 recommendations, prioritized ruthlessly. Every recommendation needs >=1 evidence_id. JSON only."""


def run(state: _common.IntelState) -> dict:
    """LLM call: produce prioritized recommendations + the executive briefing."""
    sysmsg = _common.SYSTEM.format(company=state["company"], industry=state["industry"])
    analysis = json.dumps({"opportunities": state.get("opportunities", []),
                           "risks": state.get("risks", []),
                           "trends": state.get("trends", [])}, ensure_ascii=False)
    user = ADVISOR_PROMPT.format(company=state["company"], analysis=analysis)
    data = _common.parse_json(
        _common.invoke([{"role": "system", "content": sysmsg},
                        {"role": "user", "content": user}],
                       temperature=state.get("temperature", cfg.TEMPERATURE)),
        {"recommendations": [], "ceo_briefing": {}})
    brief = data.get("ceo_briefing", {})
    note = " ".join(brief.get(k, "") for k in ("what_happened", "why_it_matters", "what_to_do_next"))
    return {"recommendations": data.get("recommendations", []),
            "ceo_briefing": brief,
            "messages": [AIMessage(content=note)]}
