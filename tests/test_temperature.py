"""
test_temperature.py — sweep sampling TEMPERATURE and compare, lab-style.

This mirrors the NLP-lab methodology referenced for the assignment: hold the
prompt fixed, vary ONE parameter (here, the LLM sampling temperature), run, and
compare the outputs side by side. It both asserts properties AND writes a
human-readable comparison artifact (reports/temperature_sweep.txt) you can show
in the oral exam.

Requires Ollama + the configured model. Auto-skips otherwise.

Hypotheses under test:
  * temperature=0.0 is (near-)deterministic: same question -> same answer twice.
  * higher temperature increases lexical variation across repeated runs.
"""
import os

import pytest

import agents as agent
import config as cfg

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
FIXED_QUESTION = f"What is the single biggest strategic risk facing {cfg.COMPANY} right now?"

_REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _distinct_token_ratio(text: str) -> float:
    toks = [t for t in text.lower().split() if t.isalpha()]
    return (len(set(toks)) / len(toks)) if toks else 0.0


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_runs_at_each_temperature(require_ollama, require_index, temperature):
    """The agent answers the fixed question at every temperature without error."""
    answer = agent.chat(FIXED_QUESTION, temperature=temperature)
    assert isinstance(answer, str) and len(answer.strip()) > 0
    assert not answer.startswith("(LLM unavailable")


def test_temperature_zero_is_repeatable(require_ollama, require_index):
    """temperature=0.0 should give identical (or near-identical) repeated answers."""
    a = agent.chat(FIXED_QUESTION, temperature=0.0)
    b = agent.chat(FIXED_QUESTION, temperature=0.0)
    # Allow tiny nondeterminism but expect very high overlap at temp 0.
    assert a.strip()[:200] == b.strip()[:200] or a.strip() == b.strip()


def test_sweep_and_write_comparison(require_ollama, require_index):
    """Run the sweep, assert variation rises with temperature, and save the artifact."""
    os.makedirs(_REPORTS, exist_ok=True)
    rows, variation = [], {}
    for t in TEMPERATURES:
        runs = [agent.chat(FIXED_QUESTION, temperature=t) for _ in range(2)]
        # cross-run difference: 1 - char overlap proxy via set of words
        w0, w1 = set(runs[0].lower().split()), set(runs[1].lower().split())
        jaccard = len(w0 & w1) / max(len(w0 | w1), 1)
        variation[t] = 1 - jaccard
        rows.append(f"\n=== temperature={t} (cross-run difference={variation[t]:.2f}) ===\n"
                    f"--- run 1 ---\n{runs[0].strip()}\n--- run 2 ---\n{runs[1].strip()}\n")

    out = os.path.join(_REPORTS, "temperature_sweep.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"TEMPERATURE SWEEP · {cfg.COMPANY}\nQuestion: {FIXED_QUESTION}\n")
        f.write("".join(rows))
    assert os.path.exists(out)

    # Directional check: temp 1.0 should vary at least as much as temp 0.0.
    assert variation[1.0] >= variation[0.0]
