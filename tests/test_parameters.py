"""
test_parameters.py — sweep sampling parameters and compare, lab-style.

Extends the temperature sweep to the other knobs the factory now exposes:
temperature, top_p, top_k, num_predict. Same methodology as the NLP lab — hold
the prompt fixed, vary ONE parameter, run, compare. Writes a human-readable
artifact (reports/parameter_sweep.txt) for the oral exam.

These call the model directly (no retrieval), so they need Ollama but not the
index. Auto-skip when Ollama isn't reachable.

Hypotheses under test:
  * num_predict caps the number of tokens generated.
  * a fixed seed makes generation reproducible even at temperature > 0.
  * top_p / top_k / temperature change lexical variation across runs.
"""
import os

import pytest

import llm
import config as cfg

FIXED_PROMPT = (
    f"In two sentences, state the single biggest strategic risk facing {cfg.COMPANY}."
)

# one knob per row -> {param_name: [values to try]}
PARAM_GRID = {
    "temperature": [0.0, 0.5, 1.0],
    "top_p":       [0.3, 0.7, 1.0],
    "top_k":       [10, 40, 100],
    "num_predict": [32, 128],
}

_REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _gen(**params) -> str:
    msg = llm.get_llm(**params).invoke(FIXED_PROMPT)
    return msg.content if hasattr(msg, "content") else str(msg)


def _words(text: str):
    return [t for t in text.lower().split() if t.isalpha()]


@pytest.mark.parametrize("param,values", list(PARAM_GRID.items()))
def test_each_parameter_runs(require_ollama, param, values):
    """Generation succeeds for every value of every swept parameter."""
    for v in values:
        out = _gen(temperature=0.3, **{param: v})
        assert isinstance(out, str) and out.strip(), f"empty output for {param}={v}"


def test_num_predict_caps_length(require_ollama):
    """A small num_predict must produce a clearly shorter response than a large one."""
    short = _gen(temperature=0.7, seed=1, num_predict=24)
    long = _gen(temperature=0.7, seed=1, num_predict=256)
    assert len(_words(short)) < len(_words(long))


def test_seed_is_reproducible(require_ollama):
    """A fixed seed yields identical output across runs even at temperature > 0."""
    a = _gen(temperature=0.9, seed=42)
    b = _gen(temperature=0.9, seed=42)
    assert a.strip() == b.strip()


def test_sweep_and_write_artifact(require_ollama):
    """Run the full grid, save a side-by-side artifact, assert all cells produced text."""
    os.makedirs(_REPORTS, exist_ok=True)
    lines = [f"PARAMETER SWEEP · {cfg.COMPANY} · model={cfg.LLM_MODEL}",
             f"Prompt: {FIXED_PROMPT}\n"]
    for param, values in PARAM_GRID.items():
        lines.append(f"\n########## sweep: {param} ##########")
        for v in values:
            out = _gen(temperature=0.3, **{param: v}).strip()
            wc = len(_words(out))
            lines.append(f"\n--- {param}={v}  ({wc} words) ---\n{out}")
            assert out, f"empty output for {param}={v}"

    out_path = os.path.join(_REPORTS, "parameter_sweep.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    assert os.path.exists(out_path)
