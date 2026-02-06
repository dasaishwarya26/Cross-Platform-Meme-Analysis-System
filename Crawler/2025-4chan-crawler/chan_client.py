import requests
import logging
import os

# r = requests.get("http://a.4cdn.org/pol/threads.json")

# print(f"{r}")

# print(f"{r.json()}")

API_BASE_URL = "http://a.4cdn.org"

logger = logging.getLogger("4chan client")
logger.propagate = False

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)

logger.setLevel(numeric_level)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)


class ChanClient:
    def get_threads(self, board):
        api_call = self.build_request([board, "threads.json"])
        return self.execute_request(api_call)

    def get_thread(self, board, thread_number):
        api_call = self.build_request([board, "thread", f"{thread_number}.json"])
        return self.execute_request(api_call)

    def get_catalog(self, board):
        api_call = self.build_request([board, "catalog.json"])
        return self.execute_request(api_call)

    # Build the full endpoint URL.
    def build_request(self, call_pieces=[]):
        api_call = "/".join([API_BASE_URL] + call_pieces)

        return api_call

    """
    This should execute an api call, so go out and actuall do the http get
    """

    def execute_request(self, api_call):
        logger.info(f"api call: {api_call}")
        r = requests.get(api_call)
        if r.status_code == 404:
            logger.info(f"404 for {api_call}")
            return dict()

        # logger.info(f"{r.text}")
        return r.json()


if __name__ == "__main__":
    # print(f"{get_catalog("pol")}")

    # instantiate client
    client = ChanClient()

    print(f"{client.get_thread('pol', 516443305)}")
    # print(f"{get_threads("pol")}")
    # client = ChanClient()
    # json = client.get_thread("pol", 124205675)
    # print(json)
