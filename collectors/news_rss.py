"""
News collector. Pulls from Google News RSS + Yahoo Finance RSS.
No API key required. Casts multiple queries to comfortably clear 100 docs.
"""
import sys, os, json, hashlib, time
from datetime import datetime
from urllib.parse import quote_plus

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import feedparser
except ImportError:
    sys.exit("Run: pip install feedparser")


def _gnews(query):
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def build_feeds():
    c = config.COMPANY
    feeds = [
        _gnews(f"{c} stock"),
        _gnews(f"{c} earnings"),
        _gnews(f"{c} AI"),
        _gnews(f"{c} cloud"),
        _gnews(f"{c} strategy"),
        _gnews(f"{c} acquisition"),
        _gnews(f"{c} partnership"),
        _gnews(f"{c} layoffs OR restructuring"),
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={config.TICKER}&region=US&lang=en-US",
    ]
    # Competitor context feeds add breadth and let the agent reason comparatively.
    for comp in config.COMPETITORS[:3]:
        feeds.append(_gnews(f"{comp} {c}"))
    return feeds


def clean_html(text):
    import re
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collect():
    docs = []
    for url in build_feeds():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"  ! feed failed: {e}")
            continue
        for entry in feed.entries:
            link = entry.get("link", "")
            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", "")) or title
            if not title:
                continue
            docs.append({
                "id": hashlib.md5(link.encode()).hexdigest() if link else hashlib.md5(title.encode()).hexdigest(),
                "source": "news",
                "title": title,
                "text": f"{title}. {summary}",
                "url": link,
                "published": entry.get("published", datetime.utcnow().isoformat()),
            })
        time.sleep(0.4)  # be polite to the feed endpoints
    return docs


if __name__ == "__main__":
    docs = collect()
    out = os.path.join(config.DATA_DIR, "news.json")
    json.dump(docs, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[news] collected {len(docs)} docs -> {out}")
