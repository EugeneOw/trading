import logging
import random
from constants import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class StateManager:
    def __init__(self, macd_threshold, epsilon):
        self.macd_threshold = macd_threshold
        self.epsilon: float = epsilon

        self.state_map = constants.STATE_MAP
        self.avail_instrument = constants.AVAILABLE_INSTRUMENTS
        self.avail_actions = constants.AVAILABLE_ACTIONS
        self.ema_period_1 = constants.EMA_PERIODS[0]
        self.ema_period_2 = constants.EMA_PERIODS[1]

    @staticmethod
    def create_weights():
        """
        Creates a list that stores equally distributed amount of weights
        :return:
        :rtype: list[float]
        """
        number_of_instruments = len(constants.AVAILABLE_INSTRUMENTS)
        return [1/number_of_instruments]*number_of_instruments

    def define_state(self, row_content, instrument_weight):
        """
        Defines the state of each row. The list that contains the weights of each instrument is used to determine,
        which instrument is best suited for this but is not updated. It is updated only later when we calculate the
        reward.
        :param row_content: Contains details of each individual row

        :param instrument_weight: Contains weights of each individual instrument
        :type instrument_weight: list[float]

        :return: Returns a string containing action
        :rtype: str
        """
        try:
            macd = row_content[constants.AVAILABLE_INSTRUMENTS[0]]
            signal_line = row_content['Signal Line']
            ema_12 = row_content[f'{constants.AVAILABLE_INSTRUMENTS[1]} {self.ema_period_1}']
            ema_26 = row_content[f'{constants.AVAILABLE_INSTRUMENTS[1]} {self.ema_period_2}']

            # Selects state randomly
            if random.uniform(0, 1) < self.epsilon:
                return random.choice([self.macd_state(macd, signal_line), self.ema_state(ema_12, ema_26)])
            else:
                if random.uniform(0, 1) < 0.1:
                    # 10% chance of selecting the least weighted instrument.
                    highest_score = instrument_weight.index(min(instrument_weight))
                else:
                    highest_score = instrument_weight.index(max(instrument_weight))

                if highest_score == 0:
                    return self.macd_state(macd, signal_line)
                else:
                    return self.ema_state(ema_12, ema_26)

        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
            raise

    def macd_state(self, macd, signal_line):
        """
        Performs calculations and determines if current row belongs to Buy or Sell state
        :param macd: MACD value of row
        :type macd: float

        :param signal_line: Signal line value of row
        :type signal_line: float

        :return: Returns 2 parameters - Action to be taken (Buy/Sell) and instrument used.
        :return: action: Action to be taken
        :rtype: action: int

        :return: instrument: Instrument used to perform this deduction.
        :rtype: instrument: int
        """
        instrument = self.avail_instrument.index("MACD")
        if (macd > signal_line) and (macd - signal_line) < self.macd_threshold:
            action = self.state_map[self.avail_actions[0]]
        elif (macd < signal_line) and (signal_line - macd) < self.macd_threshold:
            action = self.state_map[self.avail_actions[1]]
        else:
            action = self.state_map[self.avail_actions[2]]
        return action, instrument

    def ema_state(self, ema_12, ema_26):
        """
        Performs calculations and determines if current row belongs to Buy or Sell state
        :param ema_12: MACD value of row
        :type ema_12: float

        :param ema_26: Signal line value of row
        :type ema_26: float

        :return: Returns 2 parameters - Action to be taken (Buy/Sell) and instrument used.
        :return: action: Action to be taken
        :rtype: action: int

        :return: instrument: Instrument used to perform this deduction.
        :rtype: instrument: int
        """
        instrument = self.avail_instrument.index("EMA")
        if (ema_12 - ema_26) > 0:
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
        return action, instrument
