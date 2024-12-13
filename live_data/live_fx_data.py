import os
import json
import requests
from constants import constants
from datetime import datetime, timezone
from api_key_extractor import api_key_extractor


class LiveFX:
    def __init__(self):
        self.db_file = self.get_api_key()

    @staticmethod
    def get_api_key():
        """
        Gets api key from APIKeyExtractor
        :return: api key
        :rtype: str
        """
        api_key = api_key_extractor.APIKeyExtractor()
        api_key = api_key.extract_api_key()
        return api_key


if __name__ == "__main__":
    LiveFX()
