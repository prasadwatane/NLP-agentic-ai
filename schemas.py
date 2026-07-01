"""
schemas.py — typed state + structured output, mirroring agents-from-scratch.

Two kinds of types live here:

1. AgentState (TypedDict)         — working memory passed between graph nodes.
2. Pydantic models                — the *structured terminal tool* the agent
                                    calls to finish (StrategicAnalysis) plus the
                                    nested item schemas. Binding a Pydantic model
                                    as a tool is how a tool-calling agent emits
                                    structured output: when the model decides it
                                    has gathered enough evidence, it calls
                                    StrategicAnalysis(...) and we read its
                                    validated arguments straight into analysis.json.

The field names are chosen to match exactly what the existing Streamlit
dashboard and the new headless HTML report read, so either front-end renders
the agent's output unchanged.
"""
from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------- graph state
class AgentState(TypedDict, total=False):
    """Working memory for one agent run (briefing or chat)."""
    messages: Annotated[list, add_messages]   # tool-calling loop + chat history
    company: str
    industry: str
    mode: str                                 # "briefing" | "chat"
    temperature: float                        # sampling temperature for this run
    k: int                                    # retrieval breadth (context size knob)
    analysis: dict                            # StrategicAnalysis result -> analysis.json
    answer: str                               # chat reply


# ----------------------------------------------------------------------------- evidence
class Evidence(BaseModel):
    """A single supporting snippet, copied by the model from a tool observation."""
    title: str = Field(description="Headline/title of the source document")
    source: str = Field(description="Source channel, e.g. news, hackernews, github")
    url: str = Field(default="", description="Link to the source, if available")


# ----------------------------------------------------------------------------- analysis items
class Opportunity(BaseModel):
    title: str
    impact: Literal["High", "Medium", "Low"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One sentence tying the opportunity to evidence")
    evidence: List[Evidence] = Field(default_factory=list)


class Risk(BaseModel):
    title: str
    category: Literal[
        "Competitive", "Regulatory", "Financial",
        "Operational", "Reputational", "Technology",
    ]
    severity: Literal["High", "Medium", "Low"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: List[Evidence] = Field(default_factory=list)


class Trend(BaseModel):
    title: str
    evidence: List[Evidence] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str = Field(description="Specific strategic action, not a platitude")
    priority: Literal["High", "Medium", "Low"]
    expected_impact: str = Field(description="Revenue / market / customer impact")
    risk_level: Literal["High", "Medium", "Low"]
    evidence: List[Evidence] = Field(default_factory=list)


class CEOBriefing(BaseModel):
    what_happened: str = Field(description="2-3 sentences on the situation")
    why_it_matters: str = Field(description="2-3 sentences on strategic significance")
    what_to_do_next: str = Field(description="2-3 sentences; the single most important move")


# ----------------------------------------------------------------------------- terminal tool
class StrategicAnalysis(BaseModel):
    """Final board briefing for the company.

    The agent calls this tool exactly once, when it has gathered enough evidence
    through search_corpus / get_sentiment / list_sources. Calling it ends the run;
    its validated arguments become analysis.json.
    """
    opportunities: List[Opportunity] = Field(description="3-5 prioritized opportunities")
    risks: List[Risk] = Field(description="3-5 prioritized risks")
    trends: List[Trend] = Field(description="3-5 emerging trends to monitor")
    recommendations: List[Recommendation] = Field(description="4-6 ruthlessly prioritized actions")
    ceo_briefing: CEOBriefing


def analysis_to_dict(model: StrategicAnalysis, company: str, industry: str) -> dict:
    """Serialize StrategicAnalysis to the exact dict the dashboards consume."""
    d = model.model_dump()
    d["company"] = company
    d["industry"] = industry
    return d
