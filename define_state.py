import logging


class DefineState:
    def __init__(self):
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

        self.bear_threshold = -1
        self.bull_threshold = 1
        self.ema_diff_limit = 0.5
        self.neutral_limits = 0

    def def_state(self, row, bear_threshold, bull_threshold, ema_difference):
        try:
            self.macd = row['MACD']
            self.ema_12 = row['EMA 12']
            self.ema_24 = row['EMA 24']

            self.bear_threshold = bear_threshold
            self.bull_threshold = bull_threshold
            self.ema_diff_limit = ema_difference

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
        if self.macd < self.bear_threshold:
            return "Bearish"
        elif self.macd > self.bull_threshold:
            return "Bullish"
        else:
            return "Neutral"

    def market_structure(self):
        ema_diff = abs(self.ema_12 - self.ema_24)
        if ema_diff > self.ema_diff_limit:
            if self.ema_12 > self.ema_24:
                return "Uptrend"
            elif self.ema_12 < self.ema_24:
                return "Downtrend"
            else:
                return "Sideways"
        elif ema_diff < self.ema_diff_limit:
            return "Sideways"
