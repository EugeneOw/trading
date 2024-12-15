import numpy as np
from constants import constants


class CalculateReward:
    def __init__(self, max_gradient, scaling_factor, gradient, midpoint):
        self.max_gradient = max_gradient
        self.scaling_factor = scaling_factor
        self.gradient = gradient
        self.midpoint = midpoint

        self.current_instrument = None
        self.instrument_weight = None
        self.current_instrument_score = None
        self.constant = None
        self.split_constant = None

    def calculate_reward(self, current_row_content, next_row_content, current_action, row_index, instrument_weight, current_instrument, episode):
        """
        Calculates the reward for selecting correct or wrong decisions.

        :param current_row_content: Contains initial content
        :type current_row_content: Dataframe

        :param next_row_content: Contains next row's content
        :type next_row_content: Dataframe

        :param current_action: Buy or sell
        :type current_action: string

        :param instrument_weight: List of instruments weight
        :type instrument_weight: List[float]

        :param current_instrument: Instrument selected for current row
        :type current_instrument: int

        :return: return positive or negative profit
        :rtype: float

        :param episode: Contains the number of which episode the training is currently at.
        :type episode: int

        :param row_index: Contains the index of the dataframe at which the training is currently at.
        :type row_index: int

        :return: instrument_weight: Contains list of weights allocated to instruments
        :rtype: instrument_weight: list[float]
        """

        current_price = current_row_content['Mid Price']
        next_price = next_row_content['Mid Price']

        if current_price is None or next_price is None:
            raise KeyError("Missing 'Mid Price' in on of the rows.")

        if not isinstance(current_action, str):
            raise TypeError(f"'current_action' should be a string, got {type(current_action)}")

        _buy_and_increase = current_action == constants.AVAILABLE_ACTIONS[0] and current_price < next_price
        _sell_and_decrease = current_action == constants.AVAILABLE_ACTIONS[1] and current_price > next_price
        if _sell_and_decrease or _sell_and_decrease:
            _outcome = 1
        else:
            _outcome = 0

        instrument_weight = self.adjust_reward(instrument_weight, current_instrument, _outcome, episode, row_index)

        if current_action == "Long":
            reward = current_price - next_price
        else:
            reward = next_price - current_price

        return reward, instrument_weight

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

    @property
    def split_weight(self):
        return self.constant / (len(constants.AVAILABLE_INSTRUMENTS)-1)

    @property
    def get_curr_instrument_score(self):
        return self.instrument_weight[self.current_instrument]

    def punish_main_instrument(self):
        self.instrument_weight[self.current_instrument] = (self.current_instrument_score - self.constant)

    def reward_other_instrument(self):
        _remaining_instruments = [
            idx for idx in range(len(constants.AVAILABLE_INSTRUMENTS))
            if idx != self.current_instrument
        ]
        for idx in _remaining_instruments:
            self.instrument_weight[idx] += self.split_constant

    def reward_main_instrument(self):
        self.instrument_weight[self.current_instrument] = (self.current_instrument_score + self.constant)

    def punish_other_instrument(self):
        _remaining_instruments = [
            idx for idx in range(len(constants.AVAILABLE_INSTRUMENTS))
            if idx != self.current_instrument
        ]
        for idx in _remaining_instruments:
            self.instrument_weight[idx] -= self.split_constant

    def adjust_reward(self, instrument_weight, current_instrument, outcome, episode, row_index):
        """
        Adjusts the table (instrument_weight) accordingly depending on whether it was a right (outcome: 1) or wrong
        decision (outcome:0)
        :param instrument_weight: List containing all the weights allocated to instruments
        :type instrument_weight: list[float]

        :param current_instrument: Currently selected instrument
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

        self.instrument_weight = instrument_weight
        self.current_instrument = current_instrument

        self.current_instrument_score = self.get_curr_instrument_score

        self.constant = self.dynamic_alpha(episode, row_index)

        self.split_constant = self.split_weight

        if outcome == 0:
            self.reward_other_instrument()
            self.punish_main_instrument()
        else:
            self.reward_main_instrument()
            self.punish_other_instrument()

        return instrument_weight
