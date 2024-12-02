import logging
from typing import Optional
from striprtf.striprtf import rtf_to_text


class APIKeyExtractor:
    def __init__(self):
        self.rtf_file_path = '/Users/eugen/Desktop/api.rtf'

    def extract_api_key(self) -> Optional[str]:
        """
        Extracts API key from an RTF file

        :return: Extracted API key as a string
        :rtype: str or None
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
            logging.error(f"An unexpected error occurred: {e}")
        return None

