"""
Community collector — Hacker News via the Algolia HN Search API.

Free, no API key, no approval gate (unlike Reddit's post-2025 Data API).
For a tech company this surfaces real practitioner discussion: stories AND
comments mentioning the company. This is the independent "community" source.
"""
import sys, os, json, hashlib, time, re
import urllib.request, urllib.parse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

API = "https://hn.algolia.com/api/v1/search"
UA = {"User-Agent": getattr(config, "REDDIT_USER_AGENT", "ai-ceo-intelligence/0.1")}


def clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def build_queries():
    """Company + aliases, plus a couple of competitor-context queries for breadth."""
    c = config.COMPANY
    qs = [c] + [a for a in config.COMPANY_ALIASES if a.lower() != c.lower()]
    for comp in config.COMPETITORS[:2]:
        qs.append(f"{c} {comp}")
    return list(dict.fromkeys(qs))   # dedupe, keep order


def _fetch(query, tag, hits=50):
    params = urllib.parse.urlencode({"query": query, "tags": tag, "hitsPerPage": hits})
    req = urllib.request.Request(f"{API}?{params}", headers=UA)
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    return data.get("hits", [])


def collect():
    docs = []
    for q in build_queries():
        for tag in ("story", "comment"):     # discussions live in both
            try:
                hits = _fetch(q, tag)
            except Exception as e:
                print(f"  ! HN {tag} '{q}' failed: {e}")
                continue
            for h in hits:
                oid = str(h.get("objectID", ""))
                if tag == "story":
                    title = clean_html(h.get("title") or "")
                    body = clean_html(h.get("story_text") or "")
                    text = f"{title}. {body}".strip(". ")
                    link = h.get("url") or f"https://news.ycombinator.com/item?id={oid}"
                else:  # comment
                    title = clean_html(h.get("story_title") or "")
                    text = clean_html(h.get("comment_text") or "")
                    link = f"https://news.ycombinator.com/item?id={oid}"
                if len(text) < 30:
                    continue
                docs.append({
                    "id": hashlib.md5(oid.encode()).hexdigest(),
                    "source": "hackernews",
                    "title": (title or text[:80])[:300],
                    "text": text[:2000],
                    "url": link,
                    "score": h.get("points", 0),
                    "published": h.get("created_at", datetime.now(timezone.utc).isoformat()),
                })
            time.sleep(0.4)   # be polite to the API
    return docs


if __name__ == "__main__":
    docs = collect()
    out = os.path.join(config.DATA_DIR, "hackernews.json")
    json.dump(docs, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[hackernews] collected {len(docs)} docs -> {out}")
