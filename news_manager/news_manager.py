import random
import logging
import unittest
from selenium import webdriver
from gemini_api import gemini_api
from constants import keywords as k
from constants import constants as c
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class NewsManager(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.set_page_load_timeout(c.TIME_OUT)
        self.driver.get(c.WEBPAGE)

        self.executor = ThreadPoolExecutor(max_workers=4)

        self.url = None
        self.element = None
        self.keyword: str = ""
        self.url_and_title: dict = {}
        self.articles: int = c.ARTICLES

        self.gemini = gemini_api.GeminiAPI()

    def start_web_scrap(self):
        """
        Handles the web scrapping code. Enters Google website, finds search box, enters prompts and retrieves first URL link.

        :return self.url_and_title: Contains a dictionary of the URL, title and Gemini-summarized message.
        :rtype: dict
        """
        try:
            while len(self.url_and_title) < self.articles:
                self.retrieve_keyword()
                self.element = self.driver.find_element(By.NAME, "q")
                self.enter_prompt()
                self.retrieve_link()
                WebDriverWait(self.driver, 3).until(ec.presence_of_element_located((By.ID, "search")))
                self.execute_gemini()
                self.assertNotIn("No results found.", self.driver.page_source)
                self.driver.back()
        except TimeoutException:
            logging.error("TimeoutException has occurred.")
        finally:
            self.tearDown()
            return self.url_and_title

    def retrieve_keyword(self):
        """
        Randomly retrieves keyword based on pre-defined prompts.
        :return: None
        """
        self.keys = random.choice(k.keyword())

    def enter_prompt(self):
        """
        Enters prompt and presses "enter"
        :return: None
        """
        self.element.send_keys(self.keys)
        self.element.send_keys(Keys.RETURN)

    def retrieve_link(self):
        """
        Retrieves the first a[1] URL link and saves it to self.url
        :return: None
        """
        url_link = self.driver.find_element(By.XPATH, "//div[@id='search']//a[1]")
        self.url = url_link.get_attribute("href")

    def get_response(self):
        """
        Gets response from Gemini API by passing URL link retrieved.
        :return: Returns response from Gemini API
        :rtype: Json
        """
        return self.gemini.response(self.url)

    def execute_gemini(self):
        """
        Ensures that responses that contain "Error" (which are responses that Gemini cannot summarize) are not shown.
        :return: None
        """
        future = self.executor.submit(self.get_response)
        if "Error" not in future.result().text:
            if self.url not in self.url_and_title:
                self.url_and_title[self.url] = []
            self.url_and_title[self.url].append(self.keys)
            self.url_and_title[self.url].append(future.result().text)

    def tearDown(self):
        """
        Shuts down the selenium browser.
        :return: None
        """
        self.executor.shutdown(wait=True)
        self.driver.quit()

