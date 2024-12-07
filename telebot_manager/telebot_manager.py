from api_key_extractor import api_key_extractor
import telebot
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class TeleBotManager:
    def __init__(self):
        self.api_key = self.get_api_key()
        self.polling_thread = None

    @staticmethod
    def get_api_key():
        """
        Gets api_key from APIKeyExtractor
        :return: api key
        :rtype: str
        """
        api_key = api_key_extractor.APIKeyExtractor()
        api_key = api_key.extract_api_key()
        return api_key

    def connect_tele_bot(self):
        """
        Attempts to connect to telegram bot.

        :return: Synchronous class for telegram bot.
        :rtype: Synchronous class
        """
        try:
            return telebot.TeleBot(self.api_key)
        except ValueError as e:
            logging.error("Invalid token {e]", e)
        except telebot.apihelper.ApiException as e:
            logging.error("Invalid token or connection error", e)
        except requests.exceptions.ConnectionError as e:
            logging.error("Network error:", e)


class Notifier(TeleBotManager):
    def __init__(self, t_bot, msgs):
        super().__init__()
        self.telebot = t_bot
        self.chat_id = msgs.chat.id

    def send_message(self, message):
        """
        Sends message to specific chat using Telegram bot
        :param message: The message to be sent
        :return: The result of the Telebot send_message method
        """
        return self.telebot.send_message(self.chat_id, message)

    def send_photo(self, photo, message):
        """
        Sends photo to the specific chat using Telegram bot
        :param photo: The image to be sent
        :param message: The message to be sent
        :return: The result of the Telebot send_photo method
        """
        return self.telebot.send_photo(self.chat_id, photo, caption=message)
