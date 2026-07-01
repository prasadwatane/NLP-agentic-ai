"""
Company / research collector. Pulls official press feeds and tech-industry
RSS so the repository has first-party announcements plus broader tech trends.
This is the third independent source needed for the rubric.
"""
import sys, os, json, hashlib, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import feedparser
except ImportError:
    sys.exit("Run: pip install feedparser")

# First-party + industry feeds. SAP-specific by default; swap if COMPANY changes.
PRESS_FEEDS = {
    "company_pr": [
        # SAP news center RSS (official). If your company differs, replace these.
        "https://news.sap.com/feed/",
    ],
    "tech_industry": [
        "https://techcrunch.com/feed/",
        "https://www.theregister.com/headlines.atom",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
    ],
}


def clean_html(text):
    import re
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _relevant(text):
    """Keep industry items only if they mention the company or its space."""
    blob = text.lower()
    keys = [config.COMPANY.lower()] + [a.lower() for a in config.COMPANY_ALIASES]
    keys += [c.lower() for c in config.COMPETITORS]
    keys += ["enterprise software", "erp", "cloud computing", "saas", "ai agent"]
    return any(k in blob for k in keys)


def collect():
    docs = []
    for source_label, feeds in PRESS_FEEDS.items():
        for url in feeds:
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                print(f"  ! {url} failed: {e}")
                continue
            for entry in feed.entries:
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", "")) or title
                full = f"{title}. {summary}"
                # Company PR is always kept; industry feeds get filtered.
                if source_label != "company_pr" and not _relevant(full):
                    continue
                link = entry.get("link", "")
                docs.append({
                    "id": hashlib.md5((link or title).encode()).hexdigest(),
                    "source": source_label,
                    "title": title,
                    "text": full,
                    "url": link,
                    "published": entry.get("published", datetime.utcnow().isoformat()),
                })
            time.sleep(0.4)
    return docs


if __name__ == "__main__":
    docs = collect()
    out = os.path.join(config.DATA_DIR, "company_pr.json")
    json.dump(docs, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[company/research] collected {len(docs)} docs -> {out}")
