
import ema
import bot
import math
import time
import random
import logging
import telebot
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from skopt import gp_minimize
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(message)s')
matplotlib.use('Agg')


class Main:
    states: list = [('Bullish', 'Uptrend'), ('Bullish', 'Sideways'), ('Bullish', 'Downtrend'),
                    ('Bearish', 'Uptrend'), ('Bearish', 'Sideways'), ('Bearish', 'Downtrend'),
                    ('Neutral', 'Uptrend'), ('Neutral', 'Sideways'), ('Neutral', 'Downtrend')]

    actions: list = ['Buy', 'Sell', 'Hold']
    file_path: str = '/Users/eugen/Documents/GitHub/trading/forex_data/dataset.csv'

    def __init__(self):
        self.df = self.init_data(self.file_path)
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty or could not be loaded")
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
        new_table = np.zeros((len(states), len(actions)))
        new_map = {i: i for i in range(len(states))}
        return new_table, new_map


class Sub(Main):
    iter_data = {}
    call = 1

    def __init__(self, alpha, gamma, epsilon, decay, msgs, calls):
        super().__init__()
        self.q_table = None
        self.state_to_index = None
        self.alpha = alpha  # Learning rate: How much new information affect Q-Value
        self.gamma = gamma  # Discount rate: Determines importance of future rewards
        self.epsilon = epsilon
        self.episodes = 1
        self.decay = decay
        self.actions = ["Buy", "Sell", "Hold"]
        self.msgs = msgs
        self.calls = calls

    def train(self):
        self.q_table, self.state_to_index = self.init_para(self.states, self.actions)
        total_reward = 0
        for episode in range(self.episodes):
            self.epsilon = math.exp(-episode/(self.episodes/self.decay))
            episode_reward = 0
            for idx in range(len(self.df)-810001):

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
                    current_q_value = self.q_table[state_index, action]
                    max_future_q = np.max(self.q_table[next_state_index])
                    td_target = reward + self.gamma * max_future_q
                    self.q_table[state_index, action] = current_q_value + self.alpha * (td_target - current_q_value)
                except Exception as e:
                    logging.error(f"Error processing next state at index {idx + 1}: {e} ")
            if self.calls is not None:
                logging.info(f"Episode: {episode+1}/{self.episodes} \nCalls: {self.call}/{self.calls}")
                tb.send_message(self.msgs.chat.id, f"Episode: {episode+1}/{self.episodes} \nCalls: {self.call}/{self.calls}")
            self.store_para(episode_reward)
            total_reward += episode_reward
        Sub.call += 1
        return total_reward

    def store_para(self, episode_reward):
        for key, value in {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'decay': int(self.decay),
            'objective': float(episode_reward)
        }.items():
            if key in self.iter_data:
                self.iter_data[key].append(value)
            else:
                self.iter_data[key] = [value]

    def pplot(self):
        _pplot_df = pd.DataFrame(self.iter_data)
        sns.pairplot(_pplot_df)
        plt.savefig("/Users/eugen/Downloads/pair_plot.png")
        return

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


api_key = bot.BOT().extract_api_key()
if api_key:
    tb = telebot.TeleBot(api_key)
else:
    logging.error("Failed to extract API key.")

if __name__ == "__main__":
    def objective(params, msgs, n_calls):
        alpha, gamma, epsilon, decay = params
        agent = Sub(alpha=alpha, gamma=gamma, epsilon=epsilon, decay=decay, msgs=msgs, calls=n_calls)
        return -agent.train()

    @tb.message_handler(commands=['start'])
    def main(message):
        msg = message
        start_time = time.time()
        param_space: list = [
            (0.01, 0.5),
            (0.8, 0.99),
            (0.1, 1),
            (1, 10)
        ]
        n_calls = 10
        try:
            logging.info(f"Program initiated")
            result = gp_minimize(
                lambda params: objective(params, msg, n_calls),
                dimensions=param_space,
                n_calls=n_calls,
                random_state=42)

            logging.info(f"Best parameters \n==========")
            logging.info(f"Best alpha: {result.x[0]}")
            logging.info(f"Best gamma: {result.x[1]}")
            logging.info(f"Best epsilon: {result.x[2]}")
            logging.info(f"Best decay: {int(result.x[3])}")
            logging.info(f"Time taken: {time.time()-start_time:.4f} seconds")

            best_agent = Sub(alpha=result.x[0],
                             gamma=result.x[1],
                             epsilon=result.x[2],
                             decay=result.x[3],
                             msgs=msg,
                             calls=None)
            best_agent.train()
            best_agent.pplot()

        except Exception as e:
            logging.critical(f"Fatal error in training: {e}")

    tb.infinity_polling()
