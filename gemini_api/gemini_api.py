import os
import google.generativeai as genai
from constants import constants as c
from api_key_extractor import api_key_extractor


class GeminiAPI:
    def __init__(self):
        self.api_key: str = ""

        self.retrieve_api_key()
        os.environ["API_KEY"] = self.api_key
        genai.configure(api_key=os.environ["API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def retrieve_api_key(self):
        """
        Retrieves the api key for gemini.

        :return: None
        """
        api_key = api_key_extractor.APIKeyExtractor(2)
        self.api_key = api_key.extract_api_key()[3]

    def response(self, url_link):
        """
        Generates a response based on URL link and pre-defined prompt.

        :param url_link: URL link of website
        :type url_link: str

        :return: Returns the response based on Gemini.
        :rtype: Json
        """
        response = self.model.generate_content(f"{url_link} {c.PROMPT}")
        return response

