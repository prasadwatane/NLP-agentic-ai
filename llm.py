
from functools import lru_cache

from langchain_ollama import ChatOllama

import config as cfg

# Ollama sampling options this factory exposes (all hashable -> lru_cache-safe).
_SAMPLING = ("top_p", "top_k", "num_predict", "repeat_penalty", "num_ctx", "seed")


@lru_cache(maxsize=64)
def _client(model, temperature, top_p, top_k, num_predict, repeat_penalty, num_ctx, seed):
    """Cached ChatOllama keyed by (model, temperature, *sampling params)."""
    kwargs = {"model": model, "temperature": temperature, "base_url": cfg.OLLAMA_BASE_URL}
    for name, val in zip(_SAMPLING, (top_p, top_k, num_predict, repeat_penalty, num_ctx, seed)):
        if val is not None:
            kwargs[name] = val
    return ChatOllama(**kwargs)


def _resolve(params: dict) -> dict:
    """Per-call value > configured default (configuration.py) > None (model default)."""
    return {name: params.get(name, getattr(cfg, name.upper(), None)) for name in _SAMPLING}


def get_llm(temperature: float = None, **params) -> ChatOllama:
    """Primary local model at the given temperature and optional sampling params."""
    t = cfg.TEMPERATURE if temperature is None else temperature
    r = _resolve(params)
    return _client(cfg.LLM_MODEL, t, r["top_p"], r["top_k"], r["num_predict"],
                   r["repeat_penalty"], r["num_ctx"], r["seed"])


def get_tool_llm(tools, temperature: float = None, **params):
    """Primary local model with tools bound, for the tool-calling agent."""
    return get_llm(temperature, **params).bind_tools(tools)


def get_fallback_llm(temperature: float = None, **params) -> ChatOllama:
    """Fallback model (config.LLM_FALLBACK) if the primary isn't pulled."""
    t = cfg.TEMPERATURE if temperature is None else temperature
    r = _resolve(params)
    return _client(cfg.LLM_FALLBACK, t, r["top_p"], r["top_k"], r["num_predict"],
                   r["repeat_penalty"], r["num_ctx"], r["seed"])
