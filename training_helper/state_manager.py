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
        self.avail_instr = constants.AVAIL_INSTR
        self.avail_actions = constants.AVAIL_ACTIONS
        self.ema_period_1 = constants.EMA_PERIODS[0]
        self.ema_period_2 = constants.EMA_PERIODS[1]

    @staticmethod
    def create_weights():
        """
        Creates a list that stores equally distributed amount of weights.

        :return:
        :rtype: list[float]
        """
        number_of_instruments = len(constants.AVAIL_INSTR)
        return [1/number_of_instruments]*number_of_instruments

    def define_state(self, curr_row, instr_weight, decay):
        """
        Defines the state of each row. The list that contains the weights of each instrument is used to determine,
        which instrument is best suited for this but is not updated. It is updated only later when we calculate the
        reward.

        :param curr_row: Contains details of each individual row

        :param instr_weight: Contains weights of each individual instrument
        :type instr_weight: list[float]

        :param decay: Contains the value to be larger than in order to select a pre-defined action
        :type decay: float

        :return: Returns a string containing action
        :rtype: str
        """
        try:
            macd = curr_row[constants.AVAIL_INSTR[0]]
            signal_line = curr_row['Signal Line']

            ema_12 = curr_row[f'{constants.AVAIL_INSTR[1]} {self.ema_period_1}']
            ema_26 = curr_row[f'{constants.AVAIL_INSTR[1]} {self.ema_period_2}']

            mid_price = curr_row['Mid Price']
            sma = curr_row['SMA']
            low_band = curr_row['Lower Band']

            rsi = curr_row['RSI']

            # Selects state randomly
            if random.uniform(0, 1) < decay:
                return random.choice([self.macd_state(macd, signal_line),
                                      self.ema_state(ema_12, ema_26),
                                      self.sma_state(mid_price, sma, low_band),
                                      self.rsi_state(rsi)])
            else:
                if random.uniform(0, 1) < 0.1:
                    # 10% chance of selecting the least weighted instrument.
                    highest_score = instr_weight.index(min(instr_weight))
                else:
                    highest_score = instr_weight.index(max(instr_weight))
                if highest_score == 0:
                    return self.macd_state(macd, signal_line)
                elif highest_score == 1:
                    return self.ema_state(ema_12, ema_26)
                elif highest_score == 2:
                    return self.sma_state(mid_price, sma, low_band)
                else:
                    return self.rsi_state(rsi)

        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
            raise

    def macd_state(self, macd, signal_line):
        """
        Performs calculations and determines if current row belongs to Buy or Sell state.

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
        instrument = self.avail_instr.index("MACD")
        if (macd > signal_line) and (macd - signal_line) < self.macd_threshold:
            action = self.state_map[self.avail_actions[0]]
        elif (macd < signal_line) and (signal_line - macd) < self.macd_threshold:
            action = self.state_map[self.avail_actions[1]]
        else:
            action = self.state_map[self.avail_actions[2]]
        return action, instrument

    def ema_state(self, ema_12, ema_26):
        """
        Performs calculations and determines if current row belongs to Buy or Sell state.

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
        instrument = self.avail_instr.index("EMA")
        if (ema_12 - ema_26) > 0:
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
        return action, instrument

    def sma_state(self, mid_price, sma, low_band):
        """
        Performs calculation and determines if current row belongs to Buy or Sell state.
        :param mid_price: Middle price
        :type mid_price: float

        :param sma: Simple moving average
        :type sma: float

        :param upp_band: Upper band
        :type upp_band: float

        :param low_band: Lower band
        :type low_band: float

        :return: Returns 2 parameters - Action to be taken (Buy/Sell) and instrument used.
        :return: action: Action to be taken
        :rtype: action: int

        :return: instrument: Instrument used to perform this deduction.
        :rtype: instrument: int
        """
        instrument = self.avail_instr.index("SMA")
        if mid_price > sma and mid_price > low_band:
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
        return action, instrument

    def rsi_state(self, rsi):
        """
        Performs calculation and determines if current row belongs to Buy or Sell state.

        :param rsi: Relative strength index
        :type rsi: float

        :return: Returns 2 parameters - Action to be taken (Buy/Sell) and instrument used.
        :return: action: Action to be taken
        :rtype: action: int

        :return: instrument: Instrument used to perform this deduction.
        :rtype: instrument: int
        """
        instrument = self.avail_instr.index("RSI")
        if rsi < 30:
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
        return action, instrument
