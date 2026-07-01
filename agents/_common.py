"""
_common.py — shared building blocks for the specialist agent team.

Every specialist agent (retrieval, analyst, advisor, qa) and the supervisor
import from here, so the LLM wiring, JSON parsing, evidence formatting, and the
shared graph state live in exactly one place.

Design: this mirrors the travel-agent project's `config.MODEL` + per-agent files
pattern — a small shared core, then one focused agent per file. Each specialist
makes its OWN local-model call (see llm.py), so every agent genuinely reasons
rather than running fixed logic.
"""
import json
import re
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

import config as cfg
import llm as llm_factory


# ----------------------------------------------------------------------------- shared graph state
class IntelState(TypedDict, total=False):
    """Working memory passed between the supervisor and the specialist agents."""
    messages: Annotated[list, add_messages]   # conversational memory (chat mode)
    company: str
    industry: str
    question: str
    mode: str                                 # "briefing" | "chat"
    next: str                                 # supervisor routing decision
    temperature: float
    # retrieval outputs
    evidence: list                            # flattened [{tag, source, title, url}]
    evidence_block: str                       # numbered [E#] snippets for prompting
    index_map: dict                           # tag -> {title, source, url}
    search_plan: list                         # queries the RetrievalAgent decided to run
    # analyst outputs
    opportunities: list
    risks: list
    trends: list
    # advisor outputs
    recommendations: list
    ceo_briefing: dict
    # qa output
    answer: str


# ----------------------------------------------------------------------------- LLM invoke (local model)
def invoke(messages, temperature=None, json_mode=True) -> str:
    """Call the primary local model; fall back to the configured fallback model.

    Every specialist routes its LLM call through here, so the "no commercial API
    in the reasoning path" constraint is enforced in one place.
    """
    t = cfg.TEMPERATURE if temperature is None else temperature
    try:
        model = llm_factory.get_llm(t)
        if json_mode:
            # ChatOllama honors format="json"; bind it for structured turns.
            model = model.bind(format="json")
        return model.invoke(messages).content
    except Exception as e:
        try:
            model = llm_factory.get_fallback_llm(t)
            if json_mode:
                model = model.bind(format="json")
            return model.invoke(messages).content
        except Exception as e2:
            return json.dumps({"_error": f"both models failed: {e}; {e2}"})


def parse_json(raw: str, fallback):
    """Markdown/preamble-tolerant JSON parse for small local models."""
    try:
        raw = re.sub(r"^```(json)?", "", raw.strip()).strip()
        raw = re.sub(r"```$", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])
    except Exception:
        return fallback


# ----------------------------------------------------------------------------- evidence helpers
def build_evidence_block(evidence: dict):
    """Flatten themed evidence into numbered [E#] snippets the model can cite."""
    lines, index_map, n = [], {}, 1
    for theme, hits in evidence.items():
        lines.append(f"\n## Theme: {theme}")
        for h in hits:
            tag, meta = f"E{n}", h["meta"]
            snippet = h["text"][:280].replace("\n", " ")
            lines.append(f"[{tag}] ({meta['source']}) {meta['title'][:90]} :: {snippet}")
            index_map[tag] = {"title": meta["title"], "source": meta["source"], "url": meta["url"]}
            n += 1
    return "\n".join(lines), index_map


def attach_evidence(items, index_map):
    """Resolve evidence_ids -> full {title, source, url} records for the dashboard."""
    for item in items:
        item["evidence"] = [
            index_map.get(i, {"title": i, "source": "?", "url": ""})
            for i in item.get("evidence_ids", [])
        ]
    return items


# ----------------------------------------------------------------------------- shared persona
SYSTEM = (
    "You are part of the AI CEO strategic-intelligence team advising the executive "
    "board of {company} ({industry}). You reason like a McKinsey partner: "
    "evidence-driven, decisive, and honest about uncertainty. You convert "
    "information into prioritized strategic action — you never merely summarize."
)
