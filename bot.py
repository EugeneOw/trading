from striprtf.striprtf import rtf_to_text
import logging

class BOT:
    def __init__(self):
        self.rtf_file_path = '/Users/eugen/Desktop/api.rtf'

    def extract_api_key(self):
        try:
            with open(self.rtf_file_path, 'r') as file:
                rtf_content = file.read()
            return str(rtf_to_text(rtf_content)).strip()
        except Exception as e:
            logging.error("Error reading API key: {e}")
            return None