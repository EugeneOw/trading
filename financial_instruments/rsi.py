import logging
import pandas as pd
from constants import constants as c
from financial_instruments import sma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class RSI:
    def __init__(self):
        sma_handler = sma.SMA()
        self.dataset = sma_handler.calculate_bollinger_bands()

    def calculate_rsi(self):
        try:
            delta = self.dataset['Mid Price'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=c.RSI_PERIODS[0]).mean()
            avg_loss = loss.rolling(window=c.RSI_PERIODS[0]).mean()
            rs = avg_gain/avg_loss
            self.dataset['RSI'] = 100 - (100 / (1 + rs))
            return self.dataset
        except (AttributeError or KeyError):
            logging.error("Attempting to access non-existent column.")
        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except ValueError:
            logging.error(f"Error during dataset calculation.")
