import ema
import random
import numpy as np
import math
import time
from skopt import BayesSearchCV, gp_minimize
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Main:
    states: list = [('Bullish', 'Uptrend'), ('Bullish', 'Sideways'), ('Bullish', 'Downtrend'),
                    ('Bearish', 'Uptrend'), ('Bearish', 'Sideways'), ('Bearish', 'Downtrend'),
                    ('Neutral', 'Uptrend'), ('Neutral', 'Sideways'), ('Neutral', 'Downtrend')]

    actions: list = ['Buy', 'Sell', 'Hold']
    file_path: str = '/Users/eugen/Downloads/dataset.csv'

    def __init__(self):
        self.df = self.init_data(self.file_path)
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty or could not be loaded")
        self.q_table, self.state_to_index = self.init_para(self.states, self.actions)
        self.executor = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def init_data(cls, file_path):
        try:
            df = ema.EMA(file_path).macd()
            return df
        except Exception as e:
            logging.error(f"Error loading dataset. Error message {e}")
            return None

    @classmethod
    def init_para(cls, states, actions):
        new_table = np.zeros((len(states), len(actions)), dtype=int)
        new_map = {i: i for i in range(len(states))}
        return new_table, new_map


class Sub(Main):
    def __init__(self, alpha, gamma, epsilon, decay):
        super().__init__()
        self.alpha = alpha  # Learning rate: How much new information affect Q-Value
        self.gamma = gamma  # Discount rate: Determines importance of future rewards
        self.epsilon = epsilon
        self.episodes = 20
        self.decay = decay
        self.actions = ["Buy", "Sell", "Hold"]

    def train(self):
        total_reward = 0
        for episode in range(self.episodes):
            self.epsilon = math.exp(-episode/(self.episodes/self.decay))
            episode_reward = 0
            for idx in range(len(self.df) - 1):
                future_state_index = self.executor.submit(self.next_row, idx)
                try:
                    current_state = self.define_state(self.df.iloc[idx])
                    state_index = self.state_to_index[current_state]
                except KeyError as e:
                    logging.error(f"Error defining current state at index {idx}: {e}")
                    continue

                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(len(self.actions)))
                else:
                    action = np.argmax(self.q_table[state_index])

                reward = self.calc_reward(idx, action)
                episode_reward += reward
                try:
                    next_state_index = future_state_index.result()
                    self.q_table[state_index, action] = self.q_table[state_index, action] + self.alpha * \
                        (reward + self.gamma * np.max(self.q_table[next_state_index]) - self.q_table[state_index, action])
                except Exception as e:
                    logging.error(f"Error processing next state at index {idx + 1}: {e} ")

            logging.info(f"\n=== Completed episode {episode + 1}/{self.episodes} ===")
            total_reward += episode_reward

        logging.info(f"Final Q-table after training: \n{self.q_table}")
        return total_reward

    @staticmethod
    def define_state(row):
        try:
            state_map = {
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

            macd = row['MACD']
            ema_12 = row['EMA 12']
            ema_24 = row['EMA 24']

            if macd > 0:
                macd_state = "Bullish"
            elif macd < 0:
                macd_state = "Bearish"
            else:
                macd_state = "Neutral"

            if ema_12 > ema_24:
                trend_state = 'Uptrend'
            elif ema_12 < ema_24:
                trend_state = 'Downtrend'
            else:
                trend_state = 'Sideways'
            return state_map[(macd_state, trend_state)]
        except KeyError as e:
            logging.error(f"Error defining state: {e}")
            raise

    def calc_reward(self, idx, action):
        try:
            current_price = self.df.iloc[idx]['Mid Price']
            next_price = self.df.iloc[idx + 1]['Mid Price']
            action = self.actions[action]
            if action == "Buy":
                return next_price - current_price
            elif action == "Sell":
                return current_price - next_price  #
            else:
                return -0.1
        except KeyError as e:
            logging.error(f"Error calculating reward at index {idx}: {e}")
            return -1

    def next_row(self, idx):
        try:
            next_state = self.define_state(self.df.iloc[idx + 1])
            return self.state_to_index[next_state]
        except Exception as e:
            logging.error(f"Error defining next state at index {idx}: {e}")
            raise


if __name__ == "__main__":
    def objective(params):
        alpha, gamma, epsilon, decay = params
        agent = Sub(alpha=alpha, gamma=gamma, epsilon=epsilon, decay=decay)
        return -agent.train()

    start_time = time.time()
    param_space: list = [(0.01, 0.5), (0.8, 0.99), (0.1, 1), (1, 10)]
    try:
        logging.info(f"Program initiated")
        result = gp_minimize(objective, dimensions=param_space, n_calls=20, random_state=42)
        logging.info(f"Best parameters: {result.x}")
        logging.info(f"Best objective value: {result.fun}")
        logging.info(f"Time taken: {time.time()-start_time:.4f}")
    except Exception as e:
        logging.critical(f"Fatal error in training: {e}")
