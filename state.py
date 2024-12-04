import logging


class State:
    def __init__(self, upper_limit, lower_limit):
        self.state_map = {
            ("Bullish", "Uptrend"): 0,
            ("Bullish", "Downtrend"): 1,
            ("Bullish", "Sideways"): 2,
            ("Bearish", "Uptrend"): 3,
            ("Bearish", "Downtrend"): 4,
            ("Bearish", "Sideways"): 5,
            ("Neutral", "Uptrend"): 6,
            ("Neutral", "Downtrend"): 7,
            ("Neutral", "Sideways"): 8,
        }
        self.macd = None
        self.ema_12 = None
        self.ema_24 = None

        self.upper_limit: float = upper_limit
        self.lower_limit: float = lower_limit

    def define_state(self, row):
        try:
            self.macd = row['MACD']
            self.ema_12 = row['EMA 12']
            self.ema_24 = row['EMA 24']

            senti_state = self.market_sentiment()
            struc_state = self.market_structure()

            return self.state_map[(senti_state, struc_state)]

        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

    def market_sentiment(self) -> str:
        if self.macd > 0:
            return "Bullish"
        elif self.macd < 0:
            return "Bearish"
        else:
            return "Neutral"

    def market_structure(self):
        if self.ema_12 > self.ema_24:
            return 'Uptrend'
        elif self.ema_12 < self.ema_24:
            return 'Downtrend'
        else:
            return 'Sideways'
