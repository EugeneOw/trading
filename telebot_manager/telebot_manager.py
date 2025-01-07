import telebot
import requests
import logging
from constants import constants
from prettytable import PrettyTable
from api_key_extractor import api_key_extractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class TeleBotManager:
    def __init__(self):
        self.api_key = self.retrieve_api_key()
        self.polling_thread = None

    @staticmethod
    def retrieve_api_key():
        """
        Gets api_key from APIKeyExtractor.

        :return: api key
        :rtype: str
        """
        api_key = api_key_extractor.APIKeyExtractor(2)
        api_key = api_key.extract_api_key()[0]
        return api_key

    def connect_tele_bot(self):
        """
        Attempts to connect to telegram bot.

        :return: Synchronous class for telegram bot.
        :rtype: Synchronous class
        """
        try:
            bot = telebot.TeleBot(self.api_key)
            bot.remove_webhook()
            return bot
        except ValueError as e:
            logging.error("Invalid token {e]", e)
        except telebot.apihelper.ApiException as e:
            logging.error("Invalid token or connection error", e)
        except requests.exceptions.ConnectionError as e:
            logging.error("Network error:", e)


class Notifier(TeleBotManager):
    def __init__(self, tbot, message):
        super().__init__()
        self.telebot = tbot
        self.chat_id = message.chat.id

    def send_message(self, message):
        """
        Sends a message to specific chat using Telegram bot.

        :param message: The message to be sent

        :return: The result of the Telebot send_message method
        """
        return self.telebot.send_message(self.chat_id, message, parse_mode="HTML", disable_web_page_preview=True)

    def send_photo(self, photo, message):
        """
        Sends a photo to the specific chat using Telegram bot.

        :param photo: The image to be sent
        :param message: The message to be sent

        :return: The result of the Telebot send_photo method
        """
        return self.telebot.send_photo(self.chat_id, photo, caption=message, parse_mode="HTML")

    def send_table(self, q_table):
        """
        Sends q_table to the specific chat using Telegram bot.

        :param q_table: The q_table to be sent

        :return: The result of the Telebot send_table method
        """
        table = PrettyTable()
        table.field_names = ['States'] + constants.AVAIL_ACTIONS
        for state, values in zip(constants.AVAIL_STATES, q_table.tolist()):
            table.add_row([state] + values)
        return self.telebot.send_message(self.chat_id, table)
