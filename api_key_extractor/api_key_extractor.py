import os
import logging
from constants import constants
from database_manager import database_manager
from striprtf.striprtf import rtf_to_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class APIKeyExtractor:
    def __init__(self, row_id):
        self.row_id = row_id
        self.db_file = os.path.abspath(constants.PATH_DB)

        file_path_handler = database_manager.FilePathManager(self.db_file)
        self.rtf_file_path = file_path_handler.fetch_file_path(self.row_id)

    def extract_api_key(self):
        """
        Extracts API key from an RTF file stored somewhere.

        :return: Extracted API key as a string
        :rtype: str
        """
        try:
            with open(self.rtf_file_path, 'r') as file:
                rtf_content = file.read()
            plain_text = rtf_to_text(rtf_content.strip())
            keys = plain_text.splitlines()
            return [key.strip() for key in keys if key.strip()]
        except FileNotFoundError:
            logging.error("File not found.")
        except PermissionError:
            logging.error("Program does not have permission.")
        except IOError:
            logging.error("Input/Output error while reading file.")
        except Exception as e:
            logging.error("An unexpected error occurred: ", e)
