import logging
import os.path
import pandas as pd
from constants import constants as c
from database_manager import database_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class FileManager:
    def __init__(self):
        self.db_file = os.path.abspath(c.PATH_DB)
        database_manager.DBManager(self.db_file)
        self.file_path_manager = database_manager.FilePathManager(self.db_file)


class EMA(FileManager):
    def __init__(self):
        super().__init__()
        self.ema_periods = c.EMA_PERIODS
        self.csv_path = self.file_path_manager.fetch_file_path(4)
        self.dataset = pd.read_csv(self.csv_path)

    def calculate_ema(self):
        """
        Performs ema calculation.

        :return: Returns an updated csv file with new rows of respective EMA periods.
        :rtype: self.dataset: Dataframe
        """
        try:
            self.dataset = self.dataset.drop(self.dataset.columns[0], axis=1)
            ask_bid_price = self.dataset['Ask']+self.dataset['Bid']
            self.dataset['Mid Price'] = ask_bid_price/2
            for ema_period in self.ema_periods:
                self.dataset[f'EMA {ema_period}'] = self.dataset['Mid Price'].ewm(span=ema_period, adjust=False).mean()
            return self.dataset
        except (AttributeError or KeyError):
            logging.error("Attempting to access non-existent column.")
        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except ValueError:
            logging.error(f"Error during dataset calculation.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
