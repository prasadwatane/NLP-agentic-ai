"""
conftest.py — shared fixtures.

Tests come in two tiers:
  * Offline tests (tools, report, schema) run anywhere — no LLM, no index needed
    beyond the data files; they skip gracefully if the index/corpus is absent.
  * LLM tests (temperature, context sweeps) require a running Ollama with the
    configured model pulled. They auto-skip when Ollama isn't reachable, so the
    suite stays green on a machine without the model while still being runnable
    for the graded demo.
"""
import os
import sys

import pytest

# Make the project root importable (config, agents, tools, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


def _index_ready() -> bool:
    return os.path.isdir(cfg.CHROMA_DIR) and bool(os.listdir(cfg.CHROMA_DIR))


@pytest.fixture(scope="session")
def ollama_available():
    return _ollama_up()


@pytest.fixture(scope="session")
def index_available():
    return _index_ready()


@pytest.fixture
def require_ollama(ollama_available):
    if not ollama_available:
        pytest.skip(f"Ollama not reachable at {cfg.OLLAMA_BASE_URL} — skipping LLM test")


@pytest.fixture
def require_index(index_available):
    if not index_available:
        pytest.skip("ChromaDB index not built — run the pipeline first")


@pytest.fixture
def sample_analysis():
    """A minimal valid analysis dict for offline tests."""
    return {
        "company": cfg.COMPANY,
        "industry": cfg.INDUSTRY,
        "_engine": "test_fixture",
        "opportunities": [{
            "title": "Embed agentic AI across the suite", "impact": "High",
            "confidence": 0.7, "rationale": "Demand signal across sources.",
            "evidence": [{"title": "SAP unveils AI copilot", "source": "company_pr", "url": "https://example.com"}],
        }],
        "risks": [{
            "title": "Cloud margin pressure", "category": "Competitive", "severity": "Medium",
            "confidence": 0.6, "rationale": "Hyperscaler competition.",
            "evidence": [{"title": "Margins squeezed", "source": "news", "url": ""}],
        }],
        "trends": [{"title": "Shift to consumption pricing", "evidence": []}],
        "recommendations": [{
            "action": "Ship a flagship agent for finance workflows", "priority": "High",
            "expected_impact": "Net-new ARR and stickiness", "risk_level": "Medium",
            "evidence": [],
        }],
        "ceo_briefing": {
            "what_happened": "AI demand is reshaping enterprise software buying.",
            "why_it_matters": "First movers capture renewal leverage.",
            "what_to_do_next": "Prioritize one flagship agent and ship it this quarter.",
        },
    }
