"""
config.py — the single source of configuration for the whole project.

Everything (collectors, processing, retrieval, sentiment, agents, tools, report,
server) imports from here. Change COMPANY and the entire pipeline retargets.
"""
import os

# ---- repo root (this file lives at the project root) ----------
ROOT = os.path.dirname(os.path.abspath(__file__))

# load .env if present (so LLM_MODEL / OLLAMA_BASE_URL / Reddit keys can be set there)
_envfile = os.path.join(ROOT, ".env")
if os.path.exists(_envfile):
    for _line in open(_envfile, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---- Target company ----------------------------------------------------------
COMPANY = "SAP"
INDUSTRY = "Enterprise Software / Cloud"
COMPANY_ALIASES = ["SAP SE", "SAP"]
TICKER = "SAP"
COMPETITORS = ["Oracle", "Salesforce", "Microsoft Dynamics", "Workday", "ServiceNow"]

# ---- Paths -------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")
ANALYSIS_CACHE = os.path.join(DATA_DIR, "analysis.json")
CHECKPOINT_DB = os.path.join(DATA_DIR, "agent_memory.sqlite")

# ---- Models ------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")        # override e.g. LLM_MODEL=qwen2.5:32b
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "llama3.1:8b")
COLLECTION_NAME = "intelligence"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---- Chunking ----------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---- Collection targets ------------------------------------------------------
MIN_DOCUMENTS = 100
MIN_SOURCES = 3

# ---- Agent / orchestration ---------------------------------------------------
BRIEFING_QUESTION = "If you were the CEO today, what would you do next and why?"
TEMPERATURE = 0.2          # default sampling temperature (swept by tests)
RETRIEVAL_K = 6            # default snippets per search (swept by tests)
MAX_TOOL_LOOPS = 8         # cap on ReAct iterations before forcing a conclusion
RECURSION_LIMIT = 25       # LangGraph safety cap
LLM_ROUTING = False        # legacy orchestrator: deterministic routing

# ---- optional sampling knobs (None = model default; overridable per get_llm) --
TOP_P = None
TOP_K = None
NUM_PREDICT = None
REPEAT_PENALTY = None
NUM_CTX = None
SEED = None

# ---- Reddit (free app: https://www.reddit.com/prefs/apps) --------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ai-ceo-intelligence/0.1")

os.makedirs(DATA_DIR, exist_ok=True)
