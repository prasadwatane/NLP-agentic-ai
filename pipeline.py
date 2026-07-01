"""
pipeline.py
───────────
End-to-end pipeline runner for the AI CEO Strategic Intelligence system.

Stages:
    1. collect    news · reddit · hackernews · company/research
    2. clean      normalize + dedup -> data/clean_corpus.json
    3. index      chunk + embed into ChromaDB
    4. sentiment  VADER (or FinBERT) -> data/sentiment.json
    5. analyze    tool-calling agent on local Qwen2.5 (Ollama) -> data/analysis.json

Run:
    python pipeline.py                  # full run with the tool-calling agent
    python pipeline.py --no-llm         # skip the LLM (deterministic placeholder)
    python pipeline.py --collect-only
    python pipeline.py --analyze-only   # re-run analysis on the existing index
    python pipeline.py --temperature 0.2 --k 6

Then view the dashboard:
    streamlit run streamlit_app.py
"""
import sys, os, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def step(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def collect_all():
    from collectors import news_rss, reddit_scraper, company_pr, hackernews
    jobs = [("news", news_rss, "news.json"),
            ("reddit", reddit_scraper, "reddit.json"),
            ("hackernews", hackernews, "hackernews.json"),
            ("company/research", company_pr, "company_pr.json")]
    for label, mod, fname in jobs:
        step(f"COLLECT: {label}")
        try:
            docs = mod.collect()
        except Exception as e:
            print(f"  ! {label} collector failed ({e}); skipping")
            docs = []
        json.dump(docs, open(os.path.join(config.DATA_DIR, fname), "w"),
                  indent=2, ensure_ascii=False)
        print(f"  -> {len(docs)} docs")


def _arg_value(args, flag, cast):
    args = list(args)
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            try:
                return cast(args[i + 1])
            except ValueError:
                return None
    return None


def run_analysis(args):
    """Run the tool-calling agent and cache analysis.json."""
    use_llm = "--no-llm" not in args
    temperature = _arg_value(args, "--temperature", float)
    k = _arg_value(args, "--k", int)

    import agents
    result = agents.run_briefing(use_llm=use_llm, temperature=temperature, k=k)
    result.setdefault("company", config.COMPANY)
    result.setdefault("industry", config.INDUSTRY)
    json.dump(result, open(config.ANALYSIS_CACHE, "w"), indent=2, ensure_ascii=False)
    print(f"  engine={result.get('_engine','?')}  ->  {config.ANALYSIS_CACHE}")
    return result


def main():
    args = sys.argv[1:]
    t0 = time.time()

    if "--analyze-only" not in args:
        collect_all()
        if "--collect-only" in args:
            print(f"\nDone (collect only) in {time.time()-t0:.0f}s")
            return

        step("CLEAN + DEDUP")
        from processing import clean
        clean.clean()

        step("EMBED + INDEX")
        from processing import embed_index
        embed_index.index()

        step("SENTIMENT")
        import sentiment
        sentiment.analyze_corpus()

    step("AI CEO ANALYSIS (tool-calling agent)")
    run_analysis(args)

    print(f"\n{'='*60}\n  PIPELINE COMPLETE in {time.time()-t0:.0f}s")
    print(f"  View the dashboard:  streamlit run streamlit_app.py\n{'='*60}")


if __name__ == "__main__":
    main()
