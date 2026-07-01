"""
Sentiment analysis. Default engine is VADER (fast, no model download).
Set USE_FINBERT=True for finance-tuned transformer scoring (heavier but
more impressive in a demo). Aggregates by source for the dashboard.
"""
import sys, os, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

USE_FINBERT = False  # flip to True to use ProsusAI/finbert


def _vader_scorer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    def score(text):
        c = analyzer.polarity_scores(text)["compound"]
        label = "positive" if c > 0.05 else "negative" if c < -0.05 else "neutral"
        return {"compound": c, "label": label}
    return score


def _finbert_scorer():
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def score(text):
        r = pipe(text[:512])[0]
        label = r["label"].lower()
        signed = r["score"] if label == "positive" else -r["score"] if label == "negative" else 0.0
        return {"compound": signed, "label": label}
    return score


def analyze_corpus():
    corpus = json.load(
        open(os.path.join(config.DATA_DIR, "clean_corpus.json"), encoding="utf-8")
    )
    score = _finbert_scorer() if USE_FINBERT else _vader_scorer()

    per_doc = []
    by_source = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0, "compound_sum": 0.0, "n": 0})

    for d in corpus:
        s = score(d["text"])
        per_doc.append({
            "title": d["title"], "source": d["source"],
            "label": s["label"], "compound": round(s["compound"], 3),
            "published": d.get("published", ""),
        })
        b = by_source[d["source"]]
        b[s["label"]] += 1
        b["compound_sum"] += s["compound"]
        b["n"] += 1

    summary = {}
    for src, b in by_source.items():
        summary[src] = {
            "positive": b["positive"], "neutral": b["neutral"], "negative": b["negative"],
            "avg_compound": round(b["compound_sum"] / max(b["n"], 1), 3),
            "total": b["n"],
        }

    overall = sum(p["compound"] for p in per_doc) / max(len(per_doc), 1)
    result = {
        "overall_compound": round(overall, 3),
        "by_source": summary,
        "documents": per_doc,
        "engine": "finbert" if USE_FINBERT else "vader",
    }
    out = os.path.join(config.DATA_DIR, "sentiment.json")
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[sentiment] overall={overall:+.3f} engine={result['engine']} -> {out}")
    return result


if __name__ == "__main__":
    analyze_corpus()
