import json
import logging
import requests
import oandapyV20
from constants import constants as c
from api_key_extractor import api_key_extractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class LiveFX:
    def __init__(self):
        self.url = "https://stream-fxpractice.oanda.com/v3/accounts"  # For practice
        self.instrument = c.LIVE_INSTR

        tokens = self.get_access_token()
        self.access_token = tokens[1]
        self.account_id = tokens[2]

        self.params = {"instruments": self.instrument}
        oandapyV20.API(access_token=self.access_token)
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    @staticmethod
    def get_access_token():
        """
        Gets access token and account id.

        :return: None
        """
        handler = api_key_extractor.APIKeyExtractor()
        return handler.extract_api_key()

    def get_stream(self):
        """
        Connects to Oanda API and get live stream FX data.

        :return: None
        """
        while True:
            try:
                with requests.get(f"{self.url}/{self.account_id}/pricing/stream",
                                  headers=self.headers, params=self.params,
                                  stream=True) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode("utf-8"))
                                return data
                    else:
                        logging.error(f"Unable to connect to stream. Status code: {response.status_code}")
                        logging.error(response.json())
                        break
            except requests.exceptions.RequestException as e:
                logging.error(f"An error occurred: {e}")
                logging.info("Reconnecting to the stream...")
                continue
