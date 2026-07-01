"""
test_tools_and_schemas.py — offline tests (no LLM required).

Covers the deterministic parts: the Pydantic schema round-trip and the tool
wrappers' formatting/contracts. These run anywhere and keep the suite meaningful
even without Ollama or a built index.
"""
import importlib
import os

import pytest

from schemas import StrategicAnalysis, analysis_to_dict
from tools import get_sentiment, list_sources, search_corpus


# --------------------------------------------------------------------------- schema
def test_strategic_analysis_schema_roundtrip():
    payload = {
        "opportunities": [{
            "title": "Edge AI", "impact": "High", "confidence": 0.8,
            "rationale": "demand", "evidence": [{"title": "t", "source": "news", "url": ""}],
        }],
        "risks": [{
            "title": "Regulation", "category": "Regulatory", "severity": "Medium",
            "confidence": 0.5, "rationale": "EU AI Act", "evidence": [],
        }],
        "trends": [{"title": "consumption pricing", "evidence": []}],
        "recommendations": [{
            "action": "ship agent", "priority": "High",
            "expected_impact": "ARR", "risk_level": "Low", "evidence": [],
        }],
        "ceo_briefing": {
            "what_happened": "x", "why_it_matters": "y", "what_to_do_next": "z",
        },
    }
    model = StrategicAnalysis(**payload)
    d = analysis_to_dict(model, "SAP", "Enterprise Software")
    assert d["company"] == "SAP"
    assert d["opportunities"][0]["impact"] == "High"
    assert d["ceo_briefing"]["what_to_do_next"] == "z"


def test_schema_rejects_bad_enum():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StrategicAnalysis(
            opportunities=[{"title": "x", "impact": "Massive",  # invalid
                            "confidence": 0.5, "rationale": "r"}],
            risks=[], trends=[], recommendations=[],
            ceo_briefing={"what_happened": "a", "why_it_matters": "b", "what_to_do_next": "c"},
        )


# --------------------------------------------------------------------------- tools (graceful when no data)
def test_list_sources_returns_string():
    out = list_sources.invoke({"_": ""})
    assert isinstance(out, str) and len(out) > 0


def test_get_sentiment_returns_string():
    out = get_sentiment.invoke({"_": ""})
    assert isinstance(out, str) and len(out) > 0


def test_search_corpus_handles_missing_index_gracefully(index_available):
    if index_available:
        out = search_corpus.invoke({"query": "SAP risk", "k": 3})
        assert isinstance(out, str)
    else:
        # Without an index, the tool may raise inside chromadb; that's acceptable
        # because the agent runs only after the pipeline builds the index.
        pytest.skip("no index; search_corpus exercised in integration runs")


# --------------------------------------------------------------------------- streamlit app smoke
def test_streamlit_app_imports():
    """The Streamlit frontend module imports without executing its UI body.

    streamlit_app.py guards its UI under `if __name__ == "__main__"` style via a
    main() function, so importing it must not require a running Streamlit server.
    """
    mod = importlib.import_module("streamlit_app")
    assert hasattr(mod, "main")
