# AI CEO — Strategic Intelligence Agent (specialist team)

A strategic-intelligence system for a target company (default **SAP**). It
collects multi-source documents, indexes them in ChromaDB, scores sentiment, and
runs a **team of specialist agents** — coordinated by a supervisor — on a
**local open model (Qwen2.5 via Ollama)** to produce an evidence-backed board
briefing, viewed in a **Streamlit dashboard** with a live CEO Q&A chat. No
commercial LLM API touches the reasoning path.

## The agent team

Each specialist is responsible for one thing, coordinated by a supervisor —
the same supervisor + sub-agents pattern as a multi-agent travel planner:

```
SupervisorAgent  (agents/orchestrator.py)
   ├─ RetrievalAgent  (agents/retrieval_agent.py)  plans queries (LLM) + retrieves evidence
   ├─ AnalystAgent    (agents/analyst_agent.py)    opportunities / risks / trends (LLM)
   ├─ AdvisorAgent    (agents/advisor_agent.py)    recommendations + CEO briefing (LLM)
   └─ QAAgent         (agents/qa_agent.py)         grounded chat answer (LLM)
```

Every specialist makes its **own** local-model call — so each genuinely reasons,
not just runs fixed logic. Shared plumbing (state, LLM invoke, JSON parsing,
evidence formatting) lives in `agents/_common.py`.

### Workflow

```
briefing : Supervisor → RetrievalAgent → AnalystAgent → AdvisorAgent → analysis.json
chat     : Supervisor → RetrievalAgent → QAAgent → answer   (stateful, per thread_id)
```

The supervisor routes **deterministically** by default (guarantees progress,
can't loop forever). Set `config.LLM_ROUTING=True` to let the LLM propose the
next step, validated against the deterministic backbone.

## Quickstart

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama serve                      # leave running

python pipeline.py                # collect → clean → index → sentiment → team analysis
streamlit run streamlit_app.py    # dashboard + live CEO Q&A
```

`pipeline.py` writes `data/analysis.json` + `data/sentiment.json`; the Streamlit
app reads them. Run the pipeline first.

## Layout

```
config.py             single source of configuration (change COMPANY to retarget)
pipeline.py           end-to-end runner
streamlit_app.py      dashboard + live CEO Q&A chat  ← the only frontend

agents/               the specialist team
  orchestrator.py       SupervisorAgent — coordinates specialists, owns the graph
  retrieval_agent.py    RetrievalAgent  — plans queries + retrieves
  analyst_agent.py      AnalystAgent    — opportunities / risks / trends
  advisor_agent.py      AdvisorAgent    — recommendations + CEO briefing
  qa_agent.py           QAAgent         — grounded chat answer
  _common.py            shared state, LLM invoke, JSON parse, evidence helpers

tools/                search_corpus · get_sentiment · list_sources (wrap the data layer)
collectors/           news_rss · reddit_scraper · hackernews · company_pr
processing/           clean · chunk · embed_index (ChromaDB)
retrieval.py          semantic + hybrid search
sentiment.py          corpus sentiment (VADER default; FinBERT optional)
schemas.py            Pydantic models documenting the analysis.json shape
prompts.py            prompt text
llm.py                local Ollama model factory (temperature + sampling knobs)
tests/                temperature / context / parameter sweeps + offline tests
```

## How each specialist works (for the oral exam)

- **RetrievalAgent** — calls the LLM to *plan* 4 search queries covering growth,
  risk, and trends, then runs each via `search_corpus` (hybrid retrieval over
  ChromaDB). This is RAG with an agentic planning step. Falls back to a themed
  sweep if planning yields nothing.
- **AnalystAgent** — given the evidence block, one LLM call extracts cited
  opportunities, risks, and trends as structured JSON.
- **AdvisorAgent** — given the analyst's signals, one LLM call produces
  prioritized recommendations (with expected impact + risk level) and the CEO
  briefing.
- **QAAgent** — retrieves for the question, then answers grounded in the evidence
  and prior conversation; powers the dashboard chat with stateful memory.
- **SupervisorAgent** — a LangGraph graph routing between the specialists; each
  reports back to the supervisor, which decides the next step.

## Tests — temperature & context sweeps

```bash
pytest                                  # all tests
pytest tests/test_temperature.py -v     # sweep sampling temperature
pytest tests/test_context.py -v         # sweep retrieval k
pytest tests/test_parameters.py -v      # sweep top_p / top_k / num_predict / seed
```

LLM-dependent tests auto-skip when Ollama isn't reachable; offline schema/tool
tests always run.

## Notes

- Reasoning uses only a local open model — no commercial LLM API.
- A 7B model is run sequentially across four specialist calls per briefing; the
  briefing is slower than a single call but each agent's role is explicit. If the
  team produces nothing usable, a deterministic placeholder keeps the dashboard
  rendering.
- `data/` and `chroma_db/` are built by `pipeline.py`; they ship empty.
