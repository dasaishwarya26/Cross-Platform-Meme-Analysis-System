import requests
import psycopg2
import json
import time
from datetime import datetime

# Boards to crawl
BOARDS = ["pol", "b", "gif"]

# Database connection settings (matches your Docker container)
DB_CONFIG = {
    "host": "timescaledb",
    "dbname": "memes",
    "user": "postgres",
    "password": "AplusA",
    "port": 5432
}

def get_catalog(board):
    """Fetch active threads on a board."""
    url = f"http://a.4cdn.org/{board}/catalog.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def get_thread(board, thread_no):
    """Fetch all posts in a thread."""
    url = f"http://a.4cdn.org/{board}/thread/{thread_no}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def save_post(conn, board, thread_no, post):
    """Insert a post into the TimescaleDB posts table."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO posts (board_name, thread_number, post_number, created_at, data)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, (
                board,
                thread_no,
                post["no"],
                datetime.utcfromtimestamp(post["time"]),
                json.dumps(post)
            ))
        conn.commit()
    except Exception as e:
        print(f"[!] Error saving post {post.get('no')}: {e}")

def crawl_board(board, conn):
    """Crawl one entire 4chan board and insert its posts."""
    print(f"[*] Crawling board /{board}/ ...")
    catalog = get_catalog(board)
    for page in catalog:
        for thread in page["threads"]:
            thread_no = thread["no"]
            try:
                thread_data = get_thread(board, thread_no)
                for post in thread_data["posts"]:
                    save_post(conn, board, thread_no, post)
            except Exception as e:
                print(f"[!] Error on thread {thread_no}: {e}")

def main():
    """Main loop: connect to DB and repeatedly crawl all boards."""
    print("[*] Connecting to TimescaleDB...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("[+] Connected successfully!")

    while True:
        for board in BOARDS:
            crawl_board(board, conn)
        print("[*] Sleeping for 10 minutes...\n")
        time.sleep(600)  # 10 minutes

if __name__ == "__main__":
    main()
