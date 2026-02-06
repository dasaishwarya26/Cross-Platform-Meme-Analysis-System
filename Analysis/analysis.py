import os, re, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import requests
from sqlalchemy import create_engine, text


# ------------------ Rate validation helpers ------------------
from sqlalchemy import text

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


# Expected rates from the paper
EXPECTED_4CHAN_HOURLY = {
    "pol": 700,   # ~16,800/day
    "b":   500,   # ~12,000/day
    "gif": 200,   # ~4,800/day
}

EXPECTED_REDDIT_WEEKLY = {
    "memes":     15000,
    "dankmemes": 8000,
    "funny":     20000,
}

RATE_WINDOW_DAYS = 7  # use the last 7 days to estimate real rates


def validate_4chan_rates(days=RATE_WINDOW_DAYS):
    """
    Use dblink (same as your fetch_4chan_texts) to compute
    avg posts/hour, then derive day/week and compare to the
    numbers in the paper.
    """
    boards_list = "', '".join(b.replace("'", "''") for b in TARGET_BOARDS)
    sql = text(f"""
        WITH posts_4 AS (
            SELECT board_name AS board,
                   created_at
            FROM dblink('dbname=memes user=postgres password=AplusA',
              $$
                SELECT board_name, created_at, data
                FROM posts
                WHERE created_at >= now() - interval '{days} days'
                  AND board_name IN ('{boards_list}')
              $$
            ) AS t(board_name text, created_at timestamptz, data jsonb)
        ),
        hourly_counts AS (
            SELECT
                board,
                date_trunc('hour', created_at) AS hour_bucket,
                COUNT(*) AS n
            FROM posts_4
            GROUP BY board, hour_bucket
        )
        SELECT board, AVG(n) AS avg_per_hour
        FROM hourly_counts
        GROUP BY board
        ORDER BY board;
    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    if df.empty:
        print("⚠️ No 4chan data found in the last", days, "days.")
        return

    df["avg_per_day"] = df["avg_per_hour"] * 24
    df["avg_per_week"] = df["avg_per_hour"] * 24 * 7

    print("\n====== 4chan Rates (last", days, "days) ======")
    for _, row in df.iterrows():
        board = row["board"]
        avg_h = row["avg_per_hour"]
        avg_d = row["avg_per_day"]
        avg_w = row["avg_per_week"]
        expected_h = EXPECTED_4CHAN_HOURLY.get(board)

        print(f"\nBoard: /{board}/")
        print(f"  avg posts/hour (observed)  : {avg_h:8.2f}")
        print(f"  avg posts/day  (observed)  : {avg_d:8.2f}")
        print(f"  avg posts/week (observed)  : {avg_w:8.2f}")

        if expected_h is not None:
            print(f"  expected posts/hour (paper): {expected_h:8.2f}")
            ratio = avg_h / expected_h if expected_h > 0 else float('nan')
            print(f"  observed / expected        : {ratio:8.3f}")
        else:
            print("  (no expected value configured for this board)")


def validate_reddit_rates(days=RATE_WINDOW_DAYS):
    """
    Use public.posts to compute avg posts/day and week for each subreddit,
    then compare to the numbers in the paper.
    """
    sql = text("""
        WITH posts_r AS (
            SELECT subreddit,
                   created_utc
            FROM public.posts
            WHERE subreddit = ANY(:subs)
              AND created_utc >= now() - interval :days
        ),
        daily_counts AS (
            SELECT
                subreddit,
                date_trunc('day', created_utc) AS day_bucket,
                COUNT(*) AS n
            FROM posts_r
            GROUP BY subreddit, day_bucket
        )
        SELECT subreddit, AVG(n) AS avg_per_day
        FROM daily_counts
        GROUP BY subreddit
        ORDER BY subreddit;
    """)

    params = {
        "subs": TARGET_SUBREDDITS,
        "days": f"{days} days",
    }

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        print("⚠️ No Reddit data found in the last", days, "days.")
        return

    df["avg_per_week"] = df["avg_per_day"] * 7

    print("\n====== Reddit Rates (last", days, "days) ======")
    for _, row in df.iterrows():
        sub = row["subreddit"]
        avg_d = row["avg_per_day"]
        avg_w = row["avg_per_week"]
        expected_w = EXPECTED_REDDIT_WEEKLY.get(sub)

        print(f"\nSubreddit: r/{sub}")
        print(f"  avg posts/day  (observed)  : {avg_d:8.2f}")
        print(f"  avg posts/week (observed)  : {avg_w:8.2f}")

        if expected_w is not None:
            print(f"  expected posts/week (paper): {expected_w:8.2f}")
            ratio = avg_w / expected_w if expected_w > 0 else float('nan')
            print(f"  observed / expected        : {ratio:8.3f}")
        else:
            print("  (no expected value configured for this subreddit)")


# ------------------ Run validations ------------------
if __name__ == "__main__":
    validate_4chan_rates()
    validate_reddit_rates()
