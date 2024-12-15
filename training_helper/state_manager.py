import logging
import random
import numpy as np
from constants import constants
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class StateManager:
    def __init__(self, macd_threshold, ema_difference, epsilon, max_gradient, scaling_factor, gradient, midpoint):
        self._macd_threshold = macd_threshold
        self._ema_difference = ema_difference
        self.epsilon: float = epsilon

        self.max_gradient = max_gradient
        self.scaling_factor = scaling_factor
        self.gradient = gradient
        self.midpoint = midpoint

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
                return random.choice([self.macd_state(macd, signal_line),
                                      self.ema_state(ema_12, ema_26)])
            else:
                if random.uniform(0, 1) < 0.1:
                    # Selects least weighted instrument
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
        if macd > (signal_line - self._macd_threshold):
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
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
        if (ema_12 - ema_26) > self._ema_difference:
            action = self.state_map[self.avail_actions[0]]
        else:
            action = self.state_map[self.avail_actions[1]]
        return action, instrument


    def dynamic_alpha(self, episode, row_index):
        """
        Calculates the dynamic alpha based on episode and call. Utilizes the sigmoid growth.
        :param episode: Contains the number of which episode the training is currently at.
        :type episode: int

        :param row_index: Contains the index of the dataframe at which the training is currently at.
        :type row_index: int

        :return: Returns dynamic alpha for calculating reward/punishment
        """
        _sigmoid_component = self.scaling_factor / (1 + np.exp(-self.gradient * (episode - self.midpoint)))
        _log_component = np.log(row_index + 1)
        _constant = min(_sigmoid_component * _log_component, self.max_gradient)
        return _constant

    def adjust_reward(self, instrument_weight, current_instrument, next_instrument, outcome, episode, row_index):
        """
        Adjusts the table (instrument_weight) accordingly depending on whether it was a right (outcome: 1) or wrong
        decision (outcome:0)
        :param instrument_weight: List containing all the weights allocated to instruments
        :type instrument_weight: list[float]

        :param current_instrument: Currently selected instrument
        :type current_instrument: int

        :param next_instrument: Next selected instrument
        :type current_instrument: int

        :param outcome: Outcome of action, right (outcome: 1) and wrong (outcome:0)
        :type outcome: int

        :param episode: Contains the number of which episode the training is currently at.
        :type episode: int

        :param row_index: Contains the index of the dataframe at which the training is currently at.
        :type row_index: int

        :return: instrument_weight: List containing all the weights allocated to instruments
        :rtype: list[float]
        """
        _current_instrument_score = instrument_weight[current_instrument]
        _next_instrument_score = instrument_weight[next_instrument]
        _dynamic_alpha = self.dynamic_alpha(episode, row_index)
        if outcome == 0:
            _constant = (1 - _dynamic_alpha)
        else:
            _constant = (1 + _dynamic_alpha)

        #Punish/Reward
        instrument_weight[current_instrument] = _constant * _current_instrument_score
        instrument_weight[next_instrument] = _constant * _next_instrument_score



        logging.info(instrument_weight)
        time.sleep(0.001)
        return instrument_weight
