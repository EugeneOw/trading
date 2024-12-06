import numpy as np
from constants import constants
import logging

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
        return np.zeros((len(constants.STATE_MAP), len(constants.AVAILABLE_ACTIONS)))

    def update_q_table(self, q_table, current_state_index, next_state_index, action_index, reward):
        """
        :param q_table: Updates q_table based on q-learning update rule
        :param current_state_index: Current state index
        :param next_state_index: Next state row
        :param action_index: Current choice of current_action
        :param reward: Reward
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
            logging.error("ValueError with reward, gamma, or q_table values: ",e)
        except TypeError as e:
            logging.error("TypeError: ", e)
        except AttributeError as e:
            logging.error("AttributeError: ", e)
        except OverflowError as e:
            logging.error("OverflowError during Q-table update: ", e)
