# analysis_7days.py
import os
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "AplusA")   # change if needed

# Choose targets here
TARGET_SUBREDDITS = ["memes", "dankmemes", "funny"]  # <-- pick the subreddits you want
TARGET_BOARDS     = ["b", "pol", "gif"]                # <-- pick the 4chan boards you want

WINDOW_DAYS = 14

# -----------------------------
# Connect
# -----------------------------
url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(url)

with engine.connect() as conn:
    print("✅ DB now():", conn.execute(text("SELECT now();")).scalar())

# -----------------------------
# Reddit: daily and hourly (last 7 days, specific subreddits)
# -----------------------------
q_reddit_daily = text(f"""
SELECT subreddit,
       date_trunc('day', created_utc) AS day,
       COUNT(*) AS posts
FROM public.posts
WHERE subreddit = ANY(:subs)
  AND created_utc >= now() - interval '{WINDOW_DAYS} days'
GROUP BY subreddit, day
ORDER BY day;
""")

# build the subreddit list as a quoted SQL literal list
subs_list = ", ".join("'" + s.replace("'", "''") + "'" for s in TARGET_SUBREDDITS)

q_reddit_hourly = text(f"""
WITH bounds AS (
  SELECT date_trunc('hour', now() - interval '{WINDOW_DAYS} days') AS start_ts,
         date_trunc('hour', now()) AS end_ts
),
hours AS (
  SELECT gs AS bucket
  FROM bounds b,
       generate_series(b.start_ts, b.end_ts, interval '1 hour') AS gs
),
counts AS (
  SELECT date_trunc('hour', created_utc) AS bucket,
         subreddit,
         COUNT(*) AS posts
  FROM public.posts
  WHERE subreddit IN ({subs_list})
    AND created_utc >= (SELECT start_ts FROM bounds)
  GROUP BY 1,2
)
SELECT h.bucket, s.subreddit,
       COALESCE(c.posts, 0) AS posts
FROM (VALUES {', '.join(f"('{s}')" for s in TARGET_SUBREDDITS)}) AS s(subreddit)
CROSS JOIN hours h
LEFT JOIN counts c
  ON c.bucket = h.bucket AND c.subreddit = s.subreddit
ORDER BY h.bucket, s.subreddit;
""")


with engine.connect() as conn:
    df_reddit_daily = pd.read_sql_query(q_reddit_daily, conn, params={"subs": TARGET_SUBREDDITS}, parse_dates=["day"])
    df_reddit_hourly = pd.read_sql_query(q_reddit_hourly, conn, parse_dates=["bucket"])

print("\nReddit (daily, head):")
print(df_reddit_daily.head())

# -----------------------------
# 4chan via dblink: daily and hourly (last 7 days, specific boards)
# NOTE: dblink doesn't take parameters easily; we format lists safely.
# -----------------------------
def sql_list_str(items):
    # produce a quoted, comma-separated list for SQL IN ('a','b',...)
    return ", ".join("'" + i.replace("'", "''") + "'" for i in items)

boards_list = sql_list_str(TARGET_BOARDS)

q_4chan_daily = f"""
SELECT board_name, bucket::date AS day, SUM(cnt)::bigint AS posts
FROM dblink('dbname=memes user=postgres password=AplusA',
  $$
    SELECT board_name,
           date_trunc('day', created_at) AS bucket,
           COUNT(*)::bigint AS cnt
    FROM posts
    WHERE created_at >= now() - interval '{WINDOW_DAYS} days'
      AND board_name IN ({boards_list})
    GROUP BY board_name, date_trunc('day', created_at)
  $$
) AS t(board_name text, bucket timestamptz, cnt bigint)
GROUP BY board_name, day
ORDER BY day;
"""

q_4chan_hourly = f"""
SELECT board_name, bucket, cnt AS posts
FROM dblink('dbname=memes user=postgres password=AplusA',
  $$
    SELECT board_name,
           date_trunc('hour', created_at) AS bucket,
           COUNT(*)::bigint AS cnt
    FROM posts
    WHERE created_at >= now() - interval '{WINDOW_DAYS} days'
      AND board_name IN ({boards_list})
    GROUP BY board_name, date_trunc('hour', created_at)
  $$
) AS t(board_name text, bucket timestamptz, cnt bigint)
ORDER BY bucket, board_name;
"""

with engine.connect() as conn:
    df_4chan_daily  = pd.read_sql_query(text(q_4chan_daily),  conn, parse_dates=["day"])
    df_4chan_hourly = pd.read_sql_query(text(q_4chan_hourly), conn, parse_dates=["bucket"])

print("\n4chan (daily, head):")
print(df_4chan_daily.head())

# -----------------------------
# Plots: Reddit vs 4chan (daily & hourly)
# -----------------------------
# Reddit daily
plt.figure(figsize=(10,4))
for sub, grp in df_reddit_daily.groupby("subreddit"):
    plt.plot(grp["day"], grp["posts"], label=f"r/{sub}")
plt.title(f"Reddit posts/day (last {WINDOW_DAYS} days)")
plt.xlabel("day"); plt.ylabel("posts/day"); plt.legend(); plt.tight_layout()
plt.xticks(rotation=45, ha="right")
plt.tight_layout(pad=2.0) 
plt.show()

# 4chan daily
plt.figure(figsize=(10,4))
for board, grp in df_4chan_daily.groupby("board_name"):
    plt.plot(grp["day"], grp["posts"], label=f"/{board}/")
plt.title(f"4chan posts/day (last {WINDOW_DAYS} days)")
plt.xlabel("day"); plt.ylabel("posts/day"); plt.legend(); plt.tight_layout()
plt.xticks(rotation=45, ha="right")
plt.tight_layout(pad=2.0) 
plt.show()

# Reddit hourly (stacked area)
pivot_reddit_h = df_reddit_hourly.pivot(index="bucket", columns="subreddit", values="posts").fillna(0)
pivot_reddit_h.plot.area(figsize=(10,4))
plt.title(f"Reddit posts/hour (last {WINDOW_DAYS} days)")
plt.xlabel("hour"); plt.ylabel("posts/hour"); plt.tight_layout()
plt.show()

# 4chan hourly (stacked area)
pivot_4chan_h = df_4chan_hourly.pivot(index="bucket", columns="board_name", values="posts").fillna(0)
pivot_4chan_h.plot.area(figsize=(10,4))
plt.title(f"4chan posts/hour (last {WINDOW_DAYS} days)")
plt.xlabel("hour"); plt.ylabel("posts/hour"); plt.tight_layout()
plt.show()

# -----------------------------
# Bonus: combined hourly totals (Reddit vs 4chan)
# -----------------------------
reddit_total = df_reddit_hourly.groupby("bucket")["posts"].sum().rename("reddit_posts")
fourchan_total = df_4chan_hourly.groupby("bucket")["posts"].sum().rename("4chan_posts")
combined = pd.concat([reddit_total, fourchan_total], axis=1).fillna(0)

combined.plot(figsize=(10,4))
plt.title(f"Reddit vs 4chan — posts/hour (last {WINDOW_DAYS} days)")
plt.xlabel("hour"); plt.ylabel("posts/hour"); plt.tight_layout()
plt.show()

print("\n✅ Done. Adjust TARGET_SUBREDDITS / TARGET_BOARDS and re-run as needed.")