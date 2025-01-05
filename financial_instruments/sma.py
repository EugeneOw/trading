import logging
import pandas as pd
from constants import constants as c
from financial_instruments import macd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class SMA:
    def __init__(self):
        macd_handler = macd.MACD()
        self.dataset = macd_handler.calculate_macd()
        self.sma_periods = c.SMA_PERIODS[0]

    def calculate_bbands(self):
        try:
            self.dataset['SMA'] = self.dataset['Mid Price'].rolling(window=self.sma_periods).mean()
            self.dataset['STD'] = self.dataset['Mid Price'].rolling(window=self.sma_periods).std()
            self.dataset['Upper Band'] = self.dataset['SMA'] + (2*self.dataset['STD'])
            self.dataset['Lower Band'] = self.dataset['SMA'] - (2 * self.dataset['STD'])
            return self.dataset
        except (AttributeError or KeyError):
            logging.error("Attempting to access non-existent column.")
        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except ValueError:
            logging.error(f"Error during dataset calculation.")


