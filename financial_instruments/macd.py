from financial_instruments import ema
import logging


class MACD:
    def __init__(self, file):
        self.df = ema.EMA(file).ema_calc()

    def macd(self):
        """
        Performs Moving average convergence/divergence calculation
        :return: Returns an updated csv file with 1 new row (MACD).
        :rtype: self.df: Dataframe
        """
        try:
            self.df[f'MACD'] = self.df['EMA 12'] - self.df['EMA 26']
            self.df[f'Signal Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
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
