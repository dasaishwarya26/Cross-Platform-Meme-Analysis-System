CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS posts (
    reddit_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    title TEXT,
    author TEXT,
    permalink TEXT,
    url TEXT,
    selftext TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_utc TIMESTAMPTZ,
    raw_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_posts_created ON posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts (subreddit);