import logging
import pandas as pd
from constants import constants
from financial_instruments import ema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class MACD:
    def __init__(self):
        self.dataset = ema.EMA().calculate_ema()
        self.ema_period_1 = constants.EMA_PERIODS[0]
        self.ema_period_2 = constants.EMA_PERIODS[1]

    def calculate_macd(self):
        """
        Performs Moving average convergence/divergence calculation
        :return: Returns an updated csv file with 1 new row (MACD).
        :rtype: self.dataset: Dataframe
        """
        try:
            self.dataset[f'MACD'] = self.dataset[f'EMA {self.ema_period_1}'] - self.dataset[f'EMA {self.ema_period_1}']
            self.dataset[f'Signal Line'] = (self.dataset['MACD'].ewm(
                span=9,
                adjust=False).mean())
            return self.dataset
        except (AttributeError or KeyError):
            logging.error("Attempting to access non-existent column.")
        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except ValueError:
            logging.error("Error during dataset calculation.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
