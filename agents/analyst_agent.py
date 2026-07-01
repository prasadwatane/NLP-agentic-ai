"""
analyst_agent.py — AnalystAgent (specialist).

Responsibility:
    Turn the retrieved evidence into cited strategic signals — opportunities,
    risks, and trends. Makes its own LLM call and parses the structured result.

Input  : state["evidence_block"]  (numbered [E#] snippets from RetrievalAgent)
Output : state["opportunities"], state["risks"], state["trends"]
"""
import config as cfg
from agents import _common

ANALYST_PROMPT = """From the EVIDENCE below about {company}, extract strategic signals.
Cite the snippet tags ([E#]) that support each item.

=== EVIDENCE ===
{evidence_block}
=== END EVIDENCE ===

Return ONLY this JSON object:
{{
  "opportunities": [{{"title": "...", "impact": "High|Medium|Low", "confidence": 0.0-1.0,
                      "evidence_ids": ["E1"], "rationale": "one sentence"}}],
  "risks": [{{"title": "...", "category": "Competitive|Regulatory|Financial|Operational|Reputational|Technology",
              "severity": "High|Medium|Low", "confidence": 0.0-1.0,
              "evidence_ids": ["E2"], "rationale": "one sentence"}}],
  "trends": [{{"title": "...", "evidence_ids": ["E3"]}}]
}}
Rules: 3-5 opportunities, 3-5 risks, 3-5 trends. Every item needs >=1 evidence_id. JSON only."""


def run(state: _common.IntelState) -> dict:
    """LLM call: extract opportunities / risks / trends from the evidence."""
    sysmsg = _common.SYSTEM.format(company=state["company"], industry=state["industry"])
    user = ANALYST_PROMPT.format(company=state["company"],
                                 evidence_block=state.get("evidence_block", ""))
    data = _common.parse_json(
        _common.invoke([{"role": "system", "content": sysmsg},
                        {"role": "user", "content": user}],
                       temperature=state.get("temperature", cfg.TEMPERATURE)),
        {"opportunities": [], "risks": [], "trends": []})
    return {"opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
            "trends": data.get("trends", [])}
