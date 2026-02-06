# -*- coding: utf-8 -*-
"""
4chan boards × Reddit subreddits
Semantic Drift (SBERT) + Perspective API Toxicity
- Drift = 1 - cosine_similarity(centroid(board), centroid(subreddit))
- Perspective: TOXICITY/INSULT/PROFANITY/IDENTITY_ATTACK (group means)
- Window: last 7 days
"""

import os, re, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import requests
from sqlalchemy import create_engine, text

# ------------------ Config ------------------
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "AplusA")

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "").strip()

WINDOW_DAYS = 14
TARGET_SUBREDDITS = ["memes", "dankmemes", "funny"]
TARGET_BOARDS     = ["b", "pol", "gif"]   

N_PER_GROUP = 800          
MIN_TEXT_LEN = 10          
PERSPECTIVE_QPS_DELAY = 0.25

# ------------------ DB connect ------------------
url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(url)
with engine.connect() as conn:
    print("✅ DB time:", conn.execute(text("SELECT now();")).scalar())

# ------------------ Helpers ------------------
TAG_RE = re.compile(r"<[^>]+>")
def clean_html(s: str) -> str:
    if not s: return ""
    s = TAG_RE.sub(" ", str(s))
    s = s.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def table_has_columns(schema: str, table: str, cols: List[str]) -> set:
    with engine.connect() as conn:
        dfc = pd.read_sql_query(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=:s AND table_name=:t
        """), conn, params={"s": schema, "t": table})
    have = {c.lower() for c in dfc["column_name"].tolist()}
    return {c for c in cols if c.lower() in have}

# ------------------ Data fetchers ------------------
def fetch_4chan_texts(boards: List[str], days=WINDOW_DAYS, limit_per_board=2000) -> pd.DataFrame:
    boards_list = "', '".join(b.replace("'", "''") for b in boards)
    sql = text(f"""
        SELECT board_name AS board,
               data->>'com' AS html,
               created_at
        FROM dblink('dbname=memes user=postgres password=AplusA',
          $$
            SELECT board_name, created_at, data
            FROM posts
            WHERE created_at >= now() - interval '{days} days'
              AND board_name IN ('{boards_list}')
            ORDER BY created_at DESC
          $$
        ) AS t(board_name text, created_at timestamptz, data jsonb)
        LIMIT {limit_per_board * max(1, len(boards))};
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)
    df["text"] = df["html"].fillna("").apply(clean_html)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN]
    if len(df) > N_PER_GROUP: df = df.sample(N_PER_GROUP, random_state=42)
    return df[["board", "text"]].reset_index(drop=True)

def fetch_reddit_texts(subs: List[str], days=WINDOW_DAYS, limit_per_sub=2000) -> pd.DataFrame:
    have = table_has_columns("public", "posts", ["title", "selftext"])
    if have:
        sql = text("""
            SELECT subreddit,
                COALESCE(NULLIF(title,''), NULLIF(selftext,'')) AS text,
                created_utc
            FROM public.posts
            WHERE subreddit = ANY(:subs)
            AND created_utc >= now() - interval :days
            AND (title <> '' OR selftext <> '')
            ORDER BY created_utc DESC
            LIMIT :limit;
        """)
        params = {
            "subs": subs,
            "days": f"{days} days",
            "limit": limit_per_sub * max(1, len(subs)),
        }

        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df = df[df["text"].str.len() >= MIN_TEXT_LEN]
        if len(df) > N_PER_GROUP: df = df.sample(N_PER_GROUP, random_state=42)
        return df[["subreddit", "text"]].reset_index(drop=True)
    else:
        print("⚠️ public.posts missing title/selftext，Reddit lost data")
        return pd.DataFrame(columns=["subreddit","text"])

# ------------------ Perspective API ------------------
def perspective_score(text: str, api_key: str) -> Dict[str, float]:
    if not api_key:
        return {"TOXICITY": np.nan, "INSULT": np.nan, "PROFANITY": np.nan, "IDENTITY_ATTACK": np.nan}
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    payload = {
        "comment": {"text": text[:3000]},
        "requestedAttributes": {
            "TOXICITY": {},
            "INSULT": {},
            "PROFANITY": {},
            "IDENTITY_ATTACK": {}
        },
        "doNotStore": True,
        "languages": ["en"]
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        js = r.json().get("attributeScores", {})
        def pull(k): return js.get(k, {}).get("summaryScore", {}).get("value", np.nan)
        return {
            "TOXICITY": pull("TOXICITY"),
            "INSULT": pull("INSULT"),
            "PROFANITY": pull("PROFANITY"),
            "IDENTITY_ATTACK": pull("IDENTITY_ATTACK"),
        }
    except Exception:
        return {"TOXICITY": np.nan, "INSULT": np.nan, "PROFANITY": np.nan, "IDENTITY_ATTACK": np.nan}

def score_group_toxicity(texts: List[str], api_key: str, max_items: int = 200) -> Dict[str, float]:
    if not texts:
        return {"TOXICITY": np.nan, "INSULT": np.nan, "PROFANITY": np.nan, "IDENTITY_ATTACK": np.nan}
    sample = texts if len(texts) <= max_items else random.sample(texts, max_items)
    acc = {"TOXICITY": [], "INSULT": [], "PROFANITY": [], "IDENTITY_ATTACK": []}
    for t in sample:
        sc = perspective_score(t, api_key)
        for k in acc: acc[k].append(sc[k])
        time.sleep(PERSPECTIVE_QPS_DELAY)
    return {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in acc.items()}

# ------------------ Embeddings (SBERT) ------------------
print("📦 Loading sentence-transformers model (all-MiniLM-L6-v2) ...")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

def centroid_embed(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((384,), dtype=np.float32)
    emb = model.encode(texts, normalize_embeddings=True)
    v = emb.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# ------------------ Pipeline ------------------
df_4 = fetch_4chan_texts(TARGET_BOARDS, days=WINDOW_DAYS)
df_r = fetch_reddit_texts(TARGET_SUBREDDITS, days=WINDOW_DAYS)
print(f"4chan rows: {len(df_4)}, Reddit rows: {len(df_r)}")

# group → embeddings + toxicity
board_vecs, board_tox = {}, {}
for b, grp in df_4.groupby("board"):
    texts = grp["text"].tolist()
    board_vecs[b] = centroid_embed(texts)
    board_tox[b] = score_group_toxicity(texts, PERSPECTIVE_API_KEY, max_items=200)
    print(f"/{b}/  texts={len(texts)}  tox={board_tox[b]}")

sub_vecs, sub_tox = {}, {}
for s, grp in df_r.groupby("subreddit"):
    texts = grp["text"].tolist()
    sub_vecs[s] = centroid_embed(texts)
    sub_tox[s] = score_group_toxicity(texts, PERSPECTIVE_API_KEY, max_items=200)
    print(f"r/{s}  texts={len(texts)}  tox={sub_tox[s]}")

boards = list({*TARGET_BOARDS} & set(board_vecs.keys()))
subs   = list({*TARGET_SUBREDDITS} & set(sub_vecs.keys()))

# drift matrix
drift = np.zeros((len(boards), len(subs)), dtype=np.float32)
for i, b in enumerate(boards):
    vb = board_vecs[b].reshape(1, -1)
    for j, s in enumerate(subs):
        vs = sub_vecs[s].reshape(1, -1)
        sim = float(cosine_similarity(vb, vs)[0, 0])
        drift[i, j] = 1.0 - sim

df_drift = pd.DataFrame(drift,
                        index=[f"/{b}/" for b in boards],
                        columns=[f"r/{s}" for s in subs])
print("\nSemantic Drift Matrix (lower = closer):")
print(df_drift.round(3))

# toxicity tables
df_board_tox = pd.DataFrame(board_tox).T
df_sub_tox   = pd.DataFrame(sub_tox).T
print("\n4chan board toxicity (mean):")
print(df_board_tox.round(3))
print("\nReddit subreddit toxicity (mean):")
print(df_sub_tox.round(3))

# ------------------ Plot heatmap (matplotlib) ------------------
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(drift, aspect="auto")
ax.set_xticks(np.arange(len(subs)))
ax.set_yticks(np.arange(len(boards)))
ax.set_xticklabels([f"r/{s}" for s in subs], rotation=45, ha="right")
ax.set_yticklabels([f"/{b}/" for b in boards])
ax.set_title(f"Semantic Drift (4chan → Reddit)\nSBERT (1 - cosine), last {WINDOW_DAYS} days")
ax.set_xlabel("Reddit subreddit")
ax.set_ylabel("4chan board")

for i in range(len(boards)):
    for j in range(len(subs)):
        ax.text(j, i, f"{drift[i,j]:.3f}", ha="center", va="center", color="w")

cbar = plt.colorbar(im)
cbar.set_label("Drift (lower = closer)")
plt.tight_layout()
plt.show()