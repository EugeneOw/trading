import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)
file_path: str = '/Users/eugen/Documents/GitHub/trading/forex_data/dataset.csv'


class EMA:
    def __init__(self):
        self.ema_periods = [12, 26]  # Change calculate_macd as well.
        self.dataset = pd.read_csv(file_path)

    def calculate_ema(self):
        """
        Performs EMA 12 & EMA 26 calculation.
        :return: Returns an updated csv file with new rows of respective EMA periods.
        :rtype: self.dataset: Dataframe
        """
        try:
            ask_bid_price = self.dataset['Ask']+self.dataset['Bid']
            self.dataset['Mid Price'] = ask_bid_price/2
            for ema_period in self.ema_periods:
                self.dataset[f'EMA {ema_period}'] = self.dataset['Mid Price'].ewm(
                    span=ema_period,
                    adjust=False,
                ).mean()
            return self.dataset
        except (AttributeError or KeyError):
            logging.error("Attempting to access non-existent column.")
        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except ValueError:
            logging.error(f"Error during dataset calculation.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
