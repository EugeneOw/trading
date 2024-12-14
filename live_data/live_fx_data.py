import json
import logging
import requests
import oandapyV20
from api_key_extractor import api_key_extractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class LiveFX:
    def __init__(self):
        self.url = "https://stream-fxpractice.oanda.com/v3/accounts"  # For practice
        self.instrument = "USD_JPY,EUR_USD"

        self.access_token, self.account_id = self.get_access_token()
        oandapyV20.API(access_token=self.access_token)

        self.headers = {"Authorization": f"Bearer {self.access_token}"}
        self.params = {"instruments": self.instrument}

    @staticmethod
    def get_access_token():
        """
        Gets access token and account id
        :return: tokens[1]: access token
        :rtype: tokens[1]: str

        :return: tokens[2]: account id
        :rtype: tokens[2]: str
        """
        handler = api_key_extractor.APIKeyExtractor()
        tokens = handler.extract_api_key()
        return tokens[1], tokens[2]

    def get_stream(self):
        """
        Connects to Oanda API and get live stream FX data
        :return None
        """
        while True:
            try:
                with requests.get(f"{self.url}/{self.account_id}/pricing/stream",
                                  headers=self.headers, params=self.params,
                                  stream=True) as response:
                    if response.status_code != 200:
                        logging.error(f"Unable to connect to stream. Status code: {response.status_code}")
                        logging.error(response.json())
                        break
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            logging.info(json.dumps(data, indent=4))
            except requests.exceptions.RequestException as e:
                logging.error(f"An error occurred: {e}")
                logging.info("Reconnecting to the stream...")
                continue
