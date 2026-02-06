import os
import re
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# ============================================================
# Database configuration
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "postgres")   # Reddit posts live here in public.posts
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "AplusA")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True)

app = FastAPI()


# ============================================================
# Constants for text selection
# ============================================================

WINDOW_DAYS_DEFAULT = 14
MIN_TEXT_LENGTH = 10
MAX_ROWS_PER_GROUP = 800


# ============================================================
# Text cleaning utilities
# ============================================================

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """
    Removes HTML tags, decodes a few HTML entities, and normalizes whitespace.
    """
    if not text:
        return ""
    text = TAG_RE.sub(" ", str(text))
    text = (
        text.replace("&gt;", ">")
            .replace("&lt;", "<")
            .replace("&amp;", "&")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Table column checker
# ============================================================

def table_has_columns(schema: str, table: str, cols: list[str]) -> set[str]:
    """
    Returns the subset of columns that actually exist in the given table.
    Useful when the table schema may differ between environments.
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
            """),
            conn,
            params={"schema": schema, "table": table},
        )

    existing = {c.lower() for c in df["column_name"].tolist()}
    return {c for c in cols if c.lower() in existing}


# ============================================================
# Fetch 4chan posts (via dblink to memes DB)
# ============================================================

def fetch_4chan_texts(
    boards: list[str],
    days: int = WINDOW_DAYS_DEFAULT,
    limit_per_board: int = 2000,
) -> pd.DataFrame:
    """
    Fetches recent 4chan posts from the `memes` database via dblink.
    Returns a DataFrame with columns: [board, text].
    """
    if not boards:
        return pd.DataFrame(columns=["board", "text"])

    boards_sql = "', '".join(b.replace("'", "''") for b in boards)

    # This assumes:
    #   - A database named "memes" exists
    #   - dblink extension is available
    #   - 4chan posts are stored in memes.posts
    sql = text(f"""
        SELECT board_name AS board,
               data->>'com' AS html,
               created_at
        FROM dblink('dbname=memes user=postgres password={DB_PASS}',
          $$
            SELECT board_name, created_at, data
            FROM posts
            WHERE created_at >= now() - interval '{days} days'
              AND board_name IN ('{boards_sql}')
            ORDER BY created_at DESC
          $$
        ) AS t(board_name text, created_at timestamptz, data jsonb)
        LIMIT {limit_per_board * max(1, len(boards))};
    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    df["text"] = df["html"].fillna("").apply(clean_html)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]

    if len(df) > MAX_ROWS_PER_GROUP:
        df = df.sample(MAX_ROWS_PER_GROUP, random_state=42)

    return df[["board", "text"]].reset_index(drop=True)


# ============================================================
# Fetch Reddit posts
# ============================================================

def fetch_reddit_texts(
    subreddits: list[str],
    days: int = WINDOW_DAYS_DEFAULT,
    limit_per_sub: int = 2000,
) -> pd.DataFrame:
    """
    Fetches recent Reddit posts from public.posts in the `postgres` DB.
    Returns a DataFrame with columns: [subreddit, text].
    """
    if not subreddits:
        return pd.DataFrame(columns=["subreddit", "text"])

    has_fields = table_has_columns("public", "posts", ["title", "selftext"])

    if not has_fields:
        # Schema unexpected — return empty frame.
        return pd.DataFrame(columns=["subreddit", "text"])

    sql = text("""
        SELECT subreddit,
               COALESCE(NULLIF(title, ''), NULLIF(selftext, '')) AS text,
               created_utc
        FROM public.posts
        WHERE subreddit = ANY(:subs)
          AND created_utc >= now() - interval :days
          AND (title <> '' OR selftext <> '')
        ORDER BY created_utc DESC
        LIMIT :limit;
    """)

    params = {
        "subs": subreddits,
        "days": f"{days} days",
        "limit": limit_per_sub * max(1, len(subreddits)),
    }

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]

    if len(df) > MAX_ROWS_PER_GROUP:
        df = df.sample(MAX_ROWS_PER_GROUP, random_state=42)

    return df[["subreddit", "text"]].reset_index(drop=True)


# ============================================================
# SBERT model and centroid embeddings
# ============================================================

print("Loading SBERT model: all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def centroid_embed(texts: list[str]) -> np.ndarray:
    """
    Encodes a list of texts with SBERT and returns the normalized centroid embedding.
    SBERT (all-MiniLM-L6-v2) produces 384-dimensional vectors.
    """
    if not texts:
        return np.zeros((384,), dtype=np.float32)

    embeddings = model.encode(texts, normalize_embeddings=True)
    centroid = embeddings.mean(axis=0)

    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


# ============================================================
# Pydantic models
# ============================================================

class PostItem(BaseModel):
    subreddit: str
    title: str | None = None
    selftext: str | None = None
    created_utc: str | None = None


class DriftRequest(BaseModel):
    boards: list[str]
    subreddits: list[str]
    days: int = WINDOW_DAYS_DEFAULT


class InfluenceRequest(BaseModel):
    boards: list[str] | None = None
    subreddits: list[str] | None = None
    days: int = WINDOW_DAYS_DEFAULT


class InfluenceRow(BaseModel):
    board_name: str
    subreddit: str
    influence_score: float


class InfluenceResponse(BaseModel):
    rows: list[InfluenceRow]


class TemporalRequest(BaseModel):
    board: str
    subreddit: str
    days: int = 30  # window size for temporal analysis


class TemporalPoint(BaseModel):
    board_name: str
    subreddit: str
    time: int
    lag_value: float


class TemporalResponse(BaseModel):
    points: list[TemporalPoint]


class ExplainRequest(BaseModel):
    analysis_type: str     # "drift", "influence", or "temporal"
    data: dict             # matrix or result dict sent from dashboard


class ExplainResponse(BaseModel):
    explanation: str


# ============================================================
# OpenAI client (LLM explanations)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# Basic test endpoints
# ============================================================

@app.get("/ping")
def ping():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get("/db_time")
def db_time():
    """
    Verifies DB connectivity by returning SELECT now().
    """
    with engine.connect() as conn:
        now = conn.execute(text("SELECT now();")).scalar()
    return {"db_time": str(now)}


@app.get("/sample_posts")
def sample_posts(limit: int = 5):
    """
    Returns a small sample of Reddit posts from public.posts.
    Useful for verifying that data is present and accessible.
    """
    sql = text("""
        SELECT subreddit, title, selftext, created_utc
        FROM public.posts
        ORDER BY created_utc DESC
        LIMIT :limit;
    """)

    rows: list[PostItem] = []
    with engine.connect() as conn:
        result = conn.execute(sql, {"limit": limit})
        for r in result:
            rows.append(
                PostItem(
                    subreddit=r.subreddit,
                    title=r.title,
                    selftext=r.selftext,
                    created_utc=str(r.created_utc),
                )
            )

    return {"count": len(rows), "posts": rows}


# ============================================================
# Semantic drift endpoint
# ============================================================

@app.post("/semantic_drift")
def semantic_drift(req: DriftRequest):
    """
    Computes semantic drift between 4chan boards and Reddit subreddits.

    Drift is defined as:
        1 - cosine_similarity(centroid(board_texts), centroid(subreddit_texts))

    Response:
    {
      "boards": [...],
      "subreddits": [...],
      "drift": [[...], [...], ...],
      "message": "..." (optional)
    }
    """

    # 1. Fetch raw texts
    df_4 = fetch_4chan_texts(req.boards, days=req.days)
    df_r = fetch_reddit_texts(req.subreddits, days=req.days)

    if df_4.empty or df_r.empty:
        return {
            "boards": req.boards,
            "subreddits": req.subreddits,
            "drift": [],
            "message": "Insufficient data in the selected time window."
        }

    # 2. Compute centroid embeddings for each board / subreddit
    board_vecs: dict[str, np.ndarray] = {
        board: centroid_embed(group["text"].tolist())
        for board, group in df_4.groupby("board")
    }

    subreddit_vecs: dict[str, np.ndarray] = {
        sub: centroid_embed(group["text"].tolist())
        for sub, group in df_r.groupby("subreddit")
    }

    # Preserve request order, but only for groups that have embeddings
    boards = [b for b in req.boards if b in board_vecs]
    subs = [s for s in req.subreddits if s in subreddit_vecs]

    if not boards or not subs:
        return {
            "boards": boards,
            "subreddits": subs,
            "drift": [],
            "message": "Not enough groups with valid text data."
        }

    # 3. Compute drift matrix (boards x subreddits)
    drift_matrix = np.zeros((len(boards), len(subs)), dtype=np.float32)

    for i, b in enumerate(boards):
        vb = board_vecs[b].reshape(1, -1)
        for j, s in enumerate(subs):
            vs = subreddit_vecs[s].reshape(1, -1)
            sim = float(cosine_similarity(vb, vs)[0, 0])
            drift_matrix[i, j] = 1.0 - sim

    return {
        "boards": boards,
        "subreddits": subs,
        "drift": drift_matrix.tolist()
    }


# ============================================================
# Influence matrix endpoint (board x subreddit, similarity-based)
# ============================================================

@app.post("/influence_matrix", response_model=InfluenceResponse)
def influence_matrix(req: InfluenceRequest):
    """
    Computes an influence-like matrix between 4chan boards and Reddit subreddits.
    Here we use SBERT cosine similarity between group centroids
    as a proxy for "influence strength".
    """

    # 1. If boards/subreddits not specified, use some defaults
    boards = req.boards or ["b", "pol", "gif"]
    subs = req.subreddits or ["memes", "dankmemes", "funny"]

    # 2. Fetch text
    df_4 = fetch_4chan_texts(boards, days=req.days)
    df_r = fetch_reddit_texts(subs, days=req.days)

    if df_4.empty or df_r.empty:
        return InfluenceResponse(rows=[])

    # 3. Compute centroid embeddings
    board_vecs: dict[str, np.ndarray] = {
        board: centroid_embed(group["text"].tolist())
        for board, group in df_4.groupby("board")
    }

    subreddit_vecs: dict[str, np.ndarray] = {
        sub: centroid_embed(group["text"].tolist())
        for sub, group in df_r.groupby("subreddit")
    }

    # Keep only groups that actually have embeddings
    boards_eff = [b for b in boards if b in board_vecs]
    subs_eff = [s for s in subs if s in subreddit_vecs]

    if not boards_eff or not subs_eff:
        return InfluenceResponse(rows=[])

    rows: list[InfluenceRow] = []

    for b in boards_eff:
        vb = board_vecs[b].reshape(1, -1)
        for s in subs_eff:
            vs = subreddit_vecs[s].reshape(1, -1)
            sim = float(cosine_similarity(vb, vs)[0, 0])  # similarity in [0,1]
            rows.append(
                InfluenceRow(
                    board_name=b,
                    subreddit=s,
                    influence_score=sim
                )
            )

    return InfluenceResponse(rows=rows)


# ============================================================
# Temporal co-occurrence endpoint
# ============================================================

@app.post("/temporal_cooccurrence", response_model=TemporalResponse)
def temporal_cooccurrence(req: TemporalRequest):
    """
    Computes a simple temporal co-occurrence / activity difference between
    a single 4chan board and a single subreddit over the last N days.

    For each day in the window:
      - chan_count = #posts on that 4chan board
      - reddit_count = #posts on that subreddit
      - lag_value = reddit_count - chan_count

    Response is a list of points with:
      - board_name
      - subreddit
      - time (0,1,2,... as index of day)
      - lag_value
    """

    board = req.board
    sub = req.subreddit
    days = req.days

    # ---- 4chan daily counts via dblink to memes DB ----
    sql_chan = text(f"""
        SELECT date_trunc('day', created_at) AS day,
               COUNT(*) AS chan_count
        FROM dblink('dbname=memes user=postgres password={DB_PASS}',
          $$
            SELECT board_name, created_at
            FROM posts
            WHERE created_at >= now() - interval '{days} days'
              AND board_name = '{board}'
          $$
        ) AS t(board_name text, created_at timestamptz)
        GROUP BY day
        ORDER BY day;
    """)

    # ---- Reddit daily counts ----
    sql_reddit = text("""
        SELECT date_trunc('day', created_utc) AS day,
               COUNT(*) AS reddit_count
        FROM public.posts
        WHERE created_utc >= now() - interval :days
          AND subreddit = :subreddit
        GROUP BY day
        ORDER BY day;
    """)

    with engine.connect() as conn:
        df_chan = pd.read_sql_query(sql_chan, conn)
        df_reddit = pd.read_sql_query(
            sql_reddit, conn, params={"days": f"{days} days", "subreddit": sub}
        )

    # Normalize column names
    if df_chan.empty:
        df_chan = pd.DataFrame(columns=["day", "chan_count"])
    if df_reddit.empty:
        df_reddit = pd.DataFrame(columns=["day", "reddit_count"])

    # Merge by day (outer join to keep all days that appear in either)
    df = pd.merge(
        df_chan,
        df_reddit,
        on="day",
        how="outer"
    ).sort_values("day").reset_index(drop=True)

    if df.empty:
        return TemporalResponse(points=[])

    df["chan_count"] = df["chan_count"].fillna(0)
    df["reddit_count"] = df["reddit_count"].fillna(0)

    # Define lag_value as reddit_count - chan_count
    df["lag_value"] = df["reddit_count"] - df["chan_count"]

    # Use simple integer time index for plotting
    df["time"] = range(len(df))

    points: list[TemporalPoint] = []
    for _, row in df.iterrows():
        points.append(
            TemporalPoint(
                board_name=board,
                subreddit=sub,
                time=int(row["time"]),
                lag_value=float(row["lag_value"])
            )
        )

    return TemporalResponse(points=points)


# ============================================================
# LLM Explanation Endpoint (for drift, influence, temporal)
# ============================================================

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """
    Uses GPT-4o-mini (via OpenAI API) to explain an analysis result
    in student-friendly language.
    """
    if client is None:
        return ExplainResponse(
            explanation="OPENAI_API_KEY is not set on the server; cannot generate explanation."
        )

    prompt = f"""
    You are an expert data scientist. A student generated a {req.analysis_type} analysis.
    Please explain the meaning of the results in simple, clear English.

    Here is the data they produced (as JSON-like dict):
    {req.data}

    Provide:
    - the main takeaway
    - insights about relationships or trends
    - any notable patterns
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain data science concepts clearly to students."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )

        explanation = completion.choices[0].message.content.strip()
        return ExplainResponse(explanation=explanation)

    except Exception as e:
        return ExplainResponse(explanation=f"Error generating explanation: {str(e)}")
