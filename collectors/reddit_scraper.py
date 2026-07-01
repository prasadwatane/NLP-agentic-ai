"""
Community collector. Two modes:
  1. PRAW (if REDDIT_CLIENT_ID/SECRET are set) - richer, higher limits.
  2. Public .json endpoints - zero credentials, lighter rate limits.

Falls back automatically so the pipeline never hard-fails on missing keys.
"""
import sys, os, json, hashlib, time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SUBREDDITS = ["stocks", "investing", "technology", "SAP", "ERP", "wallstreetbets"]


def _via_praw():
    import praw
    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )
    docs = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).search(config.COMPANY, limit=25):
                body = post.selftext or post.title
                docs.append({
                    "id": hashlib.md5(post.id.encode()).hexdigest(),
                    "source": "reddit",
                    "title": post.title,
                    "text": f"{post.title}. {body}"[:2000],
                    "url": f"https://reddit.com{post.permalink}",
                    "score": post.score,
                    "published": str(post.created_utc),
                })
        except Exception as e:
            print(f"  ! r/{sub} failed: {e}")
    return docs


def _via_public():
    docs = []
    headers = {"User-Agent": config.REDDIT_USER_AGENT}
    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/search.json?q={config.COMPANY}&restrict_sr=1&limit=25"
        try:
            req = urllib.request.Request(url, headers=headers)
            data = json.loads(urllib.request.urlopen(req, timeout=15).read())
            for child in data.get("data", {}).get("children", []):
                p = child["data"]
                body = p.get("selftext") or p.get("title", "")
                docs.append({
                    "id": hashlib.md5(p["id"].encode()).hexdigest(),
                    "source": "reddit",
                    "title": p.get("title", ""),
                    "text": f"{p.get('title','')}. {body}"[:2000],
                    "url": "https://reddit.com" + p.get("permalink", ""),
                    "score": p.get("score", 0),
                    "published": str(p.get("created_utc", "")),
                })
        except Exception as e:
            print(f"  ! r/{sub} (public) failed: {e}")
        time.sleep(1.0)
    return docs


def collect():
    if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET:
        try:
            return _via_praw()
        except ImportError:
            print("  praw not installed, using public endpoints")
    return _via_public()


if __name__ == "__main__":
    docs = collect()
    out = os.path.join(config.DATA_DIR, "reddit.json")
    json.dump(docs, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[reddit] collected {len(docs)} docs -> {out}")
