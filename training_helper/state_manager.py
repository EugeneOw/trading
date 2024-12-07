import logging
from constants import constants


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class StateManager:
    def __init__(self, macd_threshold, ema_difference):
        self._macd_threshold = macd_threshold
        self._ema_difference = ema_difference
        self.ema_period_1 = constants.EMA_PERIODS[0]
        self.ema_period_2 = constants.EMA_PERIODS[1]

    def define_state(self, row_content):
        """
        Defines the state of each row.

        :param row_content: Contains details of row

        :return: Returns a string containing action
        :rtype: str
        """
        try:
            macd = row_content['MACD']
            ema_12 = row_content[f'EMA {self.ema_period_1}']
            ema_26 = row_content[f'EMA {self.ema_period_2}']
            signal = row_content['Signal Line']

            if (macd > (signal-self._macd_threshold)) and (ema_12 > (ema_26 - self._macd_threshold)):
                return constants.STATE_MAP[constants.AVAILABLE_ACTIONS[0]]  # Buy
            elif (macd < (signal - self._macd_threshold)) and (ema_12 < (ema_26 - self._macd_threshold)):
                return constants.STATE_MAP[constants.AVAILABLE_ACTIONS[0]]  # Sell
            else:
                return constants.STATE_MAP[constants.AVAILABLE_ACTIONS[0]]  # Hold
        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
            raise


