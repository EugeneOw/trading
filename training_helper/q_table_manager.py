import logging
import numpy as np
from constants import constants


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class QTableManager:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def create_q_table():
        """
        Creates an empty (all-zeroes) q_table of size (available states) by (available actions).

        :return: numpy.array
        """
        return np.zeros((len(constants.STATE_MAP), len(constants.AVAIL_ACTIONS)))

    def update_q_table(self, q_table, current_state_index, next_state_index, action_index, reward):
        """
        Updates the q_table based on q-learning formula.

        :param q_table: Updates q_table based on a q-learning update rule
        :type q_table: numpy array

        :param current_state_index: Current row's state as an index (based-on-constants.STATE_MAP)
        :type current_state_index: int

        :param next_state_index: Next row's state as an index
        :type next_state_index: int

        :param action_index: Current action as an index (based on constants.AVAILABLE_ACTIONS)
        :type action_index: int

        :param reward: Reward (Positive or negative) based on whether the correct choice was made to
        :type reward: float

        :param q_table: Updated q_table based on a q-learning formula
        :rtype q_table: numpy array
        """
        try:
            current_q_value = q_table[current_state_index, action_index]
            max_future_q = np.max(q_table[next_state_index])
            td_target = reward + self.gamma * max_future_q
            q_table[current_state_index, action_index] = current_q_value + self.alpha * (td_target - current_q_value)
            return q_table
        except IndexError as e:
            logging.error(f"IndexError at state index={current_state_index}, current_action={action_index}: {e}")
        except ValueError as e:
            logging.error("ValueError with reward, gamma, or q_table values: ", e)
        except TypeError as e:
            logging.error("TypeError: ", e)
        except AttributeError as e:
            logging.error("AttributeError: ", e)
        except OverflowError as e:
            logging.error("OverflowError during Q-table update: ", e)
