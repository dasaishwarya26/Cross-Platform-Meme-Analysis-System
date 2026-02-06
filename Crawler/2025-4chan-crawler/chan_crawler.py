import datetime
from chan_client import ChanClient
import os
import time
from pyfaktory import Client, Consumer, Job, Producer
import logging

# these three lines allow psycopg to insert a dict into
# a jsonb coloumn
import psycopg2
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter

register_adapter(dict, Json)

from dotenv import load_dotenv

load_dotenv()



logger = logging.getLogger("4chan client")
logger.propagate = False

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)

logger.setLevel(numeric_level)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)


# get db url
DATABASE_URL = os.environ.get("DATABASE_URL")
FACTORY_SERVER_URL = os.environ.get("FACTORY_SERVER_URL")


def threads_list_to_thread_number(thread_list):
    thread_numbers = set()
    for page in thread_list:
        # print(f"{page['page']}")
        logger.debug(f"{page['page']}")
        for thread in page["threads"]:
            logger.debug(f"{thread['no']}")
            thread_numbers.add(thread["no"])

    return thread_numbers


"""enqueue a thread crawl job to get the posts in a thread"""


def enqueue_crawl_thread(board, thread_number):
    client = ChanClient()
    # we probably want to save teh output of get_thread somewherE (e.g., database)
    logger.info(f"Getting thread /{board}/{thread_number}")
    thread = client.get_thread(board, thread_number)
    logger.info(f"Finished getting thread /{board}/{thread_number}")
    # id BIGSERIAL NOT NULL,
    # board_name TEXT NOT NULL,
    # thread_number BIGINT NOT NULL,
    # post_number BIGINT NOT NULL,
    # created_at TIMESTAMPTZ NOT NULL,
    # data JSONB NOT NULL

    # thead is a json object, that has one field called `posts`
    # which is an array of all the posts in the thread
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()

    logger.debug(f"{thread}")
    if len(thread) == 0:
        logger.warning("Empty thread!")
        return

    for post in thread["posts"]:
        post_number = post["no"]
        created_at = post["time"]
        created_at = datetime.datetime.fromtimestamp(created_at)
        data = post
        # logger.info("wut?")

        # now we need to insert into our db
        # sql = f"INSERT INTO posts VALUES ({board}, {thread_number}, {post_number} {created_at}, {data})"
        # q = "INSERT INTO posts (board_name, thread_number, post_number, created_at, data) VALUES (%s, %s, %s, %s, %s) RETURNING id"
        q = "INSERT INTO posts (board_name, thread_number, post_number, created_at, data) VALUES (%s, %s, %s, %s, %s)"
        # q2 = f"INSERT INTO posts (board_name, thread_number, post_number, created_at, data) VALUES ({board}, {thread_number}, {post_number}, {created_at}, {data})"
        # logger.info("we got here?")
        # logger.info(q2)
        cur.execute(q, (board, thread_number, post_number, created_at, data))
        conn.commit()

    cur.close()
    conn.close()


"""enqueue a thread list carwl to get the live threads on a board"""


def enqueue_crawl_threads_listing(board, old_threads=[]):
    client = ChanClient()
    # faktory_server_url = "tcp://:password@localhost:7419"
    # find dead threads, and issue jobs to crawl them
    dead_threads = get_dead_threads(board, set(old_threads))
    logger.debug(f"{dead_threads}")

    old_threads = list(threads_list_to_thread_number(client.get_threads(board)))

    # now we neee to run this job again at some point in the future
    with Client(faktory_url=FACTORY_SERVER_URL, role="producer") as client:
        # we want the job to run at some point in the future
        run_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        run_at = run_at.isoformat()[:-7] + "Z"
        # logger.info(f"run_at = {run_at}")
        producer = Producer(client=client)
        job = Job(
            jobtype="crawl_thread_listing",
            args=(board, old_threads),
            queue="crawl-thread-listing",
            at=str(run_at),
        )
        producer.push(job)


"""Get a set of the threads that are now dead"""


def get_dead_threads(board, old_threads=set()):
    # faktory_server_url = "tcp://:password@localhost:7419"
    client = ChanClient()
    new_threads = threads_list_to_thread_number(client.get_threads(board))

    dead_threads = old_threads.difference(new_threads)

    for thread in dead_threads:
        with Client(faktory_url=FACTORY_SERVER_URL, role="producer") as client:
            producer = Producer(client=client)
            job = Job(
                jobtype="crawl_thread", args=("pol", thread), queue="crawl-thread"
            )
            producer.push(job)

    logger.debug(f"{dead_threads}")


if __name__ == "__main__":
    # client = ChanClient()
    # old_threads = threads_list_to_thread_number(client.get_threads("pol"))
    # faktory_server_url = "tcp://:password@localhost:7419"

    with Client(faktory_url=FACTORY_SERVER_URL, role="consumer") as client:
        consumer = Consumer(
            client=client,
            queues=["default", "crawl-thread", "crawl-thread-listing"],
            concurrency=3,
        )
        consumer.register("crawl_thread", enqueue_crawl_thread)
        consumer.register("crawl_thread_listing", enqueue_crawl_threads_listing)
        consumer.run()

    # print(f"we found dead threads: {dead_threads}")
    # loop until we've discovered some new thread

    # while True:
    #     print("starting iteration")

    #     dead_threads = get_dead_threads("pol", old_threads)
    #     faktory_server_url = "tcp://:password@localhost:7419"
    #     if len(dead_threads) > 0:
    #         print("found a dead thread")
    #         # we found a dead thread
    #         for thread in dead_threads:
    #             with Client(faktory_url=faktory_server_url, role="producer") as client:
    #                 producer = Producer(client=client)
    #                 job = Job(jobtype="crawl_thread", args=("pol", thread), queue="crawl-thread")
    #                 producer.push(job)

    #     time.sleep(10)
