import json
import logging
import requests
import oandapyV20
from api_key_extractor import api_key_extractor


class LiveFX:
    def __init__(self):
        self.access_token = self.get_access_token()

    @staticmethod
    def get_access_token():
        """
        Gets access_token from APIKeyExtractor
        :return: access token
        :rtype: str
        """
        handler = api_key_extractor.APIKeyExtractor()
        file_handler = handler.extract_api_key()
        access_token = file_handler[1]
        account_id = file_handler[2]
        logging.info(access_token)
        return access_token


