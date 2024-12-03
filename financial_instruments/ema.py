import logging
import pandas as pd


class EMA:
    def __init__(self, file):
        self.periods = [12, 24]
        self.df = pd.read_csv(file)

    def ema_calc(self):
        """
        Performs EMA 12 & EMA 24 calculation.
        :return: Returns an updated csv file with 2 new rows(EMA 12 and EMA 24).
        :rtype: self.df: Dataframe
        """
        try:
            self.df['Mid Price'] = (self.df['Ask'] + self.df['Bid']/2)
            for period in self.periods:
                self.df[f'EMA {period}'] = self.df['Mid Price'].ewm(span=period, adjust=False).mean()
            return self.df
        except AttributeError as e:
            logging.error(f"Attribute error: {e}")
            return None
        except KeyError as e:
            logging.error(f"Column not found: {e}")
            return None
        except ValueError as e:
            logging.error(f"Value error during calculation: {e}")
            return None
        except TypeError as e:
            logging.error(f"Type error in operation: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

