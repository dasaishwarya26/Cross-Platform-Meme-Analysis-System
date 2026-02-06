# crawler/reddit_crawler.py
import os
import time
import logging
from typing import Dict, Any, List

import requests
import psycopg2
from psycopg2.extras import Json

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("reddit_crawler")

# --- ENV ---
UA = os.getenv("REDDIT_USER_AGENT", "reddit-crawler/1.0")
CID = os.getenv("REDDIT_CLIENT_ID")
SEC = os.getenv("REDDIT_CLIENT_SECRET")
if not CID or not SEC:
    raise RuntimeError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET in environment.")

# Parse SUBREDDITS; support "all" (i.e., r/all global feed)
raw = os.getenv("SUBREDDITS", "").strip()
if raw:
    parsed = [s.strip() for s in raw.split(",") if s.strip()]
    # If user sets SUBREDDITS=all (case-insensitive), use the global feed
    if len(parsed) == 1 and parsed[0].lower() == "all":
        SUBREDDITS = ["all"]
        logger.info("Using global feed: r/all")
    else:
        # keep order and uniqueness
        seen = set()
        SUBREDDITS = []
        for s in parsed:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                SUBREDDITS.append(s)
else:
    logger.error("No subreddits configured. Please set SUBREDDITS environment variable.")
    import sys; sys.exit(1)

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "45"))

DB_CFG = {
    "host": os.getenv("PGHOST"),
    "port": os.getenv("PGPORT"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

# --- SQL ---
UPSERT_SQL = """
INSERT INTO posts
(reddit_id, subreddit, title, author, permalink, url, selftext, score, num_comments, created_utc, raw_json)
VALUES
(%(reddit_id)s, %(subreddit)s, %(title)s, %(author)s, %(permalink)s, %(url)s, %(selftext)s,
 %(score)s, %(num_comments)s, to_timestamp(%(created_utc)s), %(raw_json)s)
ON CONFLICT (reddit_id) DO UPDATE SET
  score = EXCLUDED.score,
  num_comments = EXCLUDED.num_comments,
  title = EXCLUDED.title,
  selftext = EXCLUDED.selftext,
  raw_json = EXCLUDED.raw_json;
"""

# --- OAuth token cache ---
_token = None
_token_expire_at = 0.0
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

def get_token(force_refresh: bool = False) -> str:
    """Fetch and cache OAuth token."""
    global _token, _token_expire_at
    now = time.time()
    if force_refresh or (not _token) or (now >= _token_expire_at):
        logger.info("Requesting new Reddit OAuth token…")
        r = SESSION.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(CID, SEC),
            data={"grant_type": "client_credentials"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        _token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        # refresh slightly early
        _token_expire_at = now + max(300, expires_in - 60)
        logger.info("Obtained token; expires_in=%s", expires_in)
    return _token

def fetch_new(subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch newest posts for a subreddit (supports 'all' as r/all)."""
    tk = get_token()
    url = f"https://oauth.reddit.com/r/{subreddit}/new"
    resp = SESSION.get(
        url,
        headers={"Authorization": f"bearer {tk}"},
        params={"limit": str(limit)},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 401:
        logger.warning("401 for r/%s, refreshing token and retrying once…", subreddit)
        tk = get_token(force_refresh=True)
        resp = SESSION.get(
            url,
            headers={"Authorization": f"bearer {tk}"},
            params={"limit": str(limit)},
            timeout=REQUEST_TIMEOUT,
        )
    # Basic rate limit handling (optional): retry once on 429 after Retry-After
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "5"))
        logger.warning("429 for r/%s, sleeping %ss then retrying once…", subreddit, retry_after)
        time.sleep(retry_after)
        resp = SESSION.get(
            url,
            headers={"Authorization": f"bearer {tk}"},
            params={"limit": str(limit)},
            timeout=REQUEST_TIMEOUT,
        )
    resp.raise_for_status()
    data = resp.json()
    children = data.get("data", {}).get("children", [])
    return [c["data"] for c in children if c.get("kind") == "t3"]

def normalize(p: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Reddit post into DB row format."""
    return {
        "reddit_id": p.get("name") or f"t3_{p.get('id')}",
        "subreddit": p.get("subreddit"),
        "title": p.get("title") or "",
        "author": p.get("author"),
        "permalink": "https://reddit.com" + (p.get("permalink") or ""),
        "url": p.get("url"),
        "selftext": p.get("selftext"),
        "score": int(p.get("score", 0) or 0),
        "num_comments": int(p.get("num_comments", 0) or 0),
        "created_utc": float(p.get("created_utc", 0.0) or 0.0),
        "raw_json": Json(p),
    }

def get_db():
    logger.info("Connecting to Postgres at %s:%s/%s", DB_CFG["host"], DB_CFG["port"], DB_CFG["dbname"])
    conn = psycopg2.connect(
        host=DB_CFG["host"],
        port=DB_CFG["port"],
        dbname=DB_CFG["dbname"],
        user=DB_CFG["user"],
        password=DB_CFG["password"],
        connect_timeout=10,
    )
    conn.autocommit = True
    logger.info("Connected.")
    return conn

def upsert_batch(conn, rows: List[Dict[str, Any]]) -> int:
    """Naive per-row upsert (simple and safe for modest volumes)."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        n = 0
        for r in rows:
            cur.execute(UPSERT_SQL, r)
            n += 1
        return n

def crawl_round(conn) -> int:
    total = 0
    for sub in SUBREDDITS:
        try:
            posts = fetch_new(sub, limit=50)
            rows = [normalize(p) for p in posts]
            n = upsert_batch(conn, rows)
            total += n
            logger.info("Upserted %s posts into %s.", n, sub)
        except Exception as e:
            logger.exception("Failed fetching/upserting r/%s: %s", sub, e)
    return total

def main():
    conn = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = get_db()
            changed = crawl_round(conn)
            logger.info("Round finished. Total upserts: %s. Sleeping %ss…", changed, POLL_SECONDS)
        except Exception as e:
            logger.exception("Round failed: %s", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
