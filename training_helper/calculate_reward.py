import logging

from training_helper import state_manager
from constants import constants


class CalculateReward:
    def __init__(self, macd_threshold, ema_difference, epsilon, max_gradient, scaling_factor, gradient, midpoint):
        self.state_handler = state_manager.StateManager(macd_threshold,
                                                        ema_difference,
                                                        epsilon,
                                                        max_gradient,
                                                        scaling_factor,
                                                        gradient,
                                                        midpoint)

    def calculate_reward(self, current_row_content, next_row_content, current_action, instrument_weight,
                         current_instrument, next_instrument, episode, row_index):
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

        :param next_instrument: Instrument selected for next row
        :type next_instrument: int

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

        instrument_weight = self.state_handler.adjust_reward(instrument_weight,
                                                             current_instrument,
                                                             next_instrument,
                                                             _outcome,
                                                             episode,
                                                             row_index)

        if current_action == "Long":
            reward = current_price - next_price
        else:
            reward = next_price - current_price

        return reward, instrument_weight
