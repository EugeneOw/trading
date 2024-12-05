import logging


class DefineState:
    def __init__(self):
        self.state_map = {
            "Buy": 0,
            "Sell": 1,
            "Hold": 2,
        }
        self.macd = None
        self.ema_12 = None
        self.ema_26 = None
        self.signal = None

        self.macd_threshold = 0
        self.ema_diff_limit = 0.5

    def def_state(self, row, macd_threshold, ema_difference):
        try:
            self.macd = row['MACD']
            self.ema_12 = row['EMA 12']
            self.ema_26 = row['EMA 26']
            self.signal = row['Signal Line']

            self.macd_threshold = macd_threshold
            self.ema_diff_limit = ema_difference

            if (self.macd > (self.signal - macd_threshold)) and (self.ema_12 > (self.ema_26 - macd_threshold)):
                action = "Buy"
            elif (self.macd < (self.signal - macd_threshold)) and (self.ema_12 < (self.ema_26 - macd_threshold)):
                action = "Sell"
            else:
                action = "Hold"
            return self.state_map[action]

        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise


