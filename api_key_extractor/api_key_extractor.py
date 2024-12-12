import os
import logging
from constants import constants
from striprtf.striprtf import rtf_to_text
from database_manager import database_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class APIKeyExtractor:
    def __init__(self):
        self.db_file = os.path.abspath(constants.PATH_DB)
        self.rtf_file_path = self.file_path().fetch_file_path(2)

    def file_path(self):
        """
        Initializes DBManager class
        """
        database_manager.DBManager(self.db_file)
        return database_manager.FilePathManager(self.db_file)

    def extract_api_key(self) -> str:
        """
        Extracts API key from an RTF file

        :return: Extracted API key as a string
        :rtype: str
        """
        try:
            with open(self.rtf_file_path, 'r') as file:
                rtf_content = file.read()
            return str(rtf_to_text(rtf_content)).strip()
        except FileNotFoundError:
            logging.error("File not found.")
        except PermissionError:
            logging.error("Program does not have permission.")
        except IOError:
            logging.error("Input/Output error while reading file.")
        except Exception as e:
            logging.error("An unexpected error occurred: ", e)
