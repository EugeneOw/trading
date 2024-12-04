import math
import time
import random
import logging
import define_state
import telebot
import notify
import requests
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import api_key_extractor
from telebot import TeleBot
from skopt import gp_minimize
from datetime import datetime
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from financial_instruments import macd

logging.basicConfig(level=logging.INFO, format='%(message)s')
matplotlib.use('Agg')
iterated_parameters: dict = {}  # Store iterated parameters
table = PrettyTable()

omitted_rows = 1  # Default 1, max 814490
no_of_calls = 20  # n_calls
no_of_episodes = 30  # episodes
reward_hist: list = []  # Store iterated objective
training_boundaries: list = [no_of_calls,
                             no_of_episodes, ]  # Stores training boundaries (no. of episodes, calls, etc.)


class Main:
    file_path: str = '/Users/eugen/Documents/GitHub/trading/forex_data/dataset.csv'

    def __init__(self):
        self.df = self.__init_data(self.file_path)

    @classmethod
    def __init_data(cls, file_path):
        """
        Retrieves csv from 'file_path' and performs MACD calculation.
        :param file_path: Contains directory of csv file
        :type file_path: str

        :return: df: Returns an updated csv file with MACD, EMA 12 and EMA 24.
        :rtype: df: Dataframe
        """
        try:
            df = macd.MACD(file_path)
            df = df.macd()
            return df

        except pd.errors.EmptyDataError:
            logging.error("Empty file or no data to load.")
        except pd.errors.ParserError:
            logging.error("Parsing error in the dataset.")
        except ValueError:
            logging.error("Value error while processing file")
        except KeyError as e:
            logging.error(f"Column not found: {e}")
        except PermissionError:
            logging.error("Permission error.")
        except FileNotFoundError:
            logging.error("File not found.")
        except OSError as e:
            logging.error(f"OS error: {e}")


class Sub(Main):
    states: list = [('Bullish', 'Uptrend'),
                    ('Bullish', 'Sideways'),
                    ('Bullish', 'Downtrend'),
                    ('Bearish', 'Uptrend'),
                    ('Bearish', 'Sideways'),
                    ('Bearish', 'Downtrend'),
                    ('Neutral', 'Uptrend'),
                    ('Neutral', 'Sideways'),
                    ('Neutral', 'Downtrend')]
    call = 1

    def __init__(self, params, alert, n_calls):
        super().__init__()
        self.q_table = None
        self.state_to_index = None

        self.alpha = params[0]
        self.gamma = params[1]
        self.epsilon = params[2]
        self.decay = params[3]
        self.bear_threshold = params[4]
        self.bull_threshold = params[5]
        self.ema_difference = params[6]

        self.episodes = training_boundaries[1]
        self.alert = alert
        self.available_actions = ["Buy", "Sell", "Hold"]
        self.n_calls = n_calls

    def train(self):
        self.q_table, self.state_to_index = self.initialize_parameters(self.states, self.available_actions)

        total_reward = 0
        state_class = define_state.DefineState()
        for episode in range(self.episodes):
            episode_reward = 0

            for _row_index in range(len(self.df) - omitted_rows):
                # Get the state of individual row.
                content_row = self.df.iloc[_row_index]
                current_state_index = state_class.def_state(content_row,
                                                            self.bear_threshold,
                                                            self.bull_threshold,
                                                            self.ema_difference)

                state_index = self.state_to_index[current_state_index]

                # Choosing current_action based on exploration rate (That decays exponentially)
                # 1) Randomly choosing a current_action or
                # 2) Selects best course of current_action based on Q-Table
                if random.uniform(0, 1) < self.epsilon:
                    action_index = random.choice(range(len(self.available_actions)))
                else:
                    action_index = np.argmax(self.q_table[state_index])
                current_action = self.available_actions[action_index]

                # Gets result of next row
                next_row_index = self.next_row(_row_index, state_class)

                # Calculates reward and update q-table based on q-learning formula
                reward = self.calculate_reward(_row_index, current_action)
                episode_reward += reward
                self.update_q_table(state_index, action_index, next_row_index, reward)
            # Updates message through telegram bot
            self.alert.notify(f"Episode: {episode + 1}/{self.episodes}"
                              f"\nCalls: {self.call}/{self.n_calls}")
            self.store_parameters(episode_reward)

            # Shows how objective changes as parameters and episode change
            reward_hist.append(episode_reward)
            total_reward += episode_reward
        Sub.call += 1
        self.log_q_table(alert=self.alert)
        return total_reward

    @classmethod
    def initialize_parameters(cls, states, available_actions):
        """
        Creates new table (Q-table) and map to store possible states
        :param states: Contains all the possible states based on MACD and EMA 12/24
        :type states: List(tuple(string, string)

        :param available_actions: Contains all the possible available_actions (buy, sell, hold) based on MACD and EMA 12/24
        :type available_actions: List(string)

        :return: new_table: Table of size len(state) by len(available_actions)
        :rtype: new_table: NumPy array

        :return: new_map: Map of all possible states and their respective numerical index
        :rtype: new_map: dict[int, int]
        """
        new_table = np.zeros((len(states), len(available_actions)))
        new_map = {i: i for i in range(len(states))}
        return new_table, new_map

    def calculate_decay(self, episode) -> float:
        """
        Calculates decay due to exponential decay
        :param episode: Episode number of learning
        :return: decay value
        :rtype: float
        """
        return math.exp(-episode / (self.episodes / self.decay))

    def next_row(self, _row_index, state_class):
        try:
            next_state = state_class.def_state(self.df.iloc[_row_index + 1],
                                               self.bear_threshold,
                                               self.bull_threshold,
                                               self.ema_difference)

            return self.state_to_index[next_state]
        except (IndexError, KeyError) as e:
            logging.error(f"Error at index {_row_index}: {str(e)}")
            raise  # Re-raise the exception for further handling or logging
        except Exception as e:
            logging.error(f"Unexpected error at index {_row_index}: {str(e)}")
            raise

    def calculate_reward(self, _row_index, current_action):
        """
        Calculates the reward for selecting correct or wrong decisions.

        :param _row_index: Contains initial mid-price

        :param current_action: Buy, sell or hold (Penalty 0.1)
        :type current_action: string

        :return: return positive or negative profit
        :rtype: double
        """
        current_price = self.df.iloc[_row_index]['Mid Price']
        next_price = self.df.iloc[_row_index + 1]['Mid Price']

        if current_price is None or next_price is None:
            raise KeyError("Missing 'Mid Price' in on of the rows.")

        if not isinstance(current_action, str):
            raise TypeError(f"'current_action' should be a string, got {type(current_action)}")

        if current_action == "Buy":
            return next_price - current_price
        elif current_action == "Sell":
            return current_price - next_price
        else:
            return -0.1

    def update_q_table(self, state_index, current_action, next_row_index, reward):
        """
        Updates q_table based on q-learning update rule
        :param state_index: Current state index
        :param current_action: Current choice of current_action
        :param next_row_index: Next state row
        :param reward: Reward
        """
        try:
            current_q_value = self.q_table[state_index, current_action]
            max_future_q = np.max(self.q_table[next_row_index])
            td_target = reward + self.gamma * max_future_q
            self.q_table[state_index, current_action] = current_q_value + self.alpha * (td_target - current_q_value)
        except IndexError as e:
            logging.error(f"IndexError at state index={state_index}, current_action={current_action}: {e}")
        except ValueError as e:
            logging.error(f"ValueError with reward, gamma, or q_table values: {e}")
        except TypeError as e:
            logging.error(f"TypeError: {e}")
        except AttributeError as e:
            logging.error(f"AttributeError: {e}")
        except OverflowError as e:
            logging.error(f"OverflowError during Q-table update: {e}")

    def store_parameters(self, episode_reward):
        """
        Stores tested parameters and result (episode_reward) into a dictionary to plot later-on
        :param episode_reward:
        :return:
        """
        for key, value in {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'decay': int(self.decay),
            'objective': float(episode_reward)
        }.items():
            try:
                if key in iterated_parameters:
                    iterated_parameters[key].append(value)
                else:
                    iterated_parameters[key] = [value]
            except AttributeError as e:
                logging.error(f"AttributeError: {e} - Ensure self.iterated_parameters is initialized as a dictionary.")
            except TypeError as e:
                logging.error(f"TypeError: {e} - Ensure self.iterated_parameters[key] is a list before calling append.")
            except Exception as e:
                logging.error(f"Unexpected error occurred while updating iterated_parameters for {key}: {e}")

    def log_q_table(self, alert):
        pretty_q_table = pd.DataFrame(self.q_table,
                                      columns=[f"{i}" for i in self.available_actions],
                                      index=self.states)
        alert.notify(f"{pretty_q_table}")


if __name__ == "__main__":
    def connect_to_telebot(key):
        """
        Attempts to connect to telegram bot.

        :param key: String containing api key
        :type key: obj: str

        :return: Synchronous class for telegram bot.
        :rtype: key: class:
        """

        try:
            return telebot.TeleBot(key)
        except ValueError:
            logging.error("Invalid token")
        except telebot.apihelper.ApiException:
            logging.error("Invalid token or connection error")
        except requests.exceptions.ConnectionError as e:
            logging.error("Network error:", e)


    def objective(params, alert, n_calls):
        """
        :param alert: Contain telegram message handler
        :param params: Contains all the parameters (i.e. alpha, ...)
        :param n_calls: Contain number of calls to perform
        :type n_calls: str or int

        :return: reward: We return a negative value as minimization is preferred rather than maximization
        """
        agent = Sub(params,
                    alert=alert,
                    n_calls=n_calls)
        reward = agent.train()
        return -reward


    def summary_update(result, alert):
        """
        Sends best parameters and images (pair-plot & line-graph) to telegram
        """
        graph_name_lst: list = ["pair_plot", "line_plot"]
        message_map: dict = {
            "alpha ": result.x[0],
            "gamma ": result.x[1],
            "epsilon ": result.x[2],
            "decay ": int(result.x[3]),
            "bearish threshold ": result.x[4],
            "bullish threshold ": result.x[5],
            "ema difference ": result.x[6]
        }

        for key, value in message_map.items():
            alert.notify(f"Best {key}:{value}")

        create_line_plot()
        create_pair_plot()

        for path in graph_name_lst:
            image_path = f"/Users/eugen/Downloads/{path}.png"
            try:
                with open(image_path, 'rb') as photo:
                    alert.notify_image(photo, f"{path}")
            except FileNotFoundError or Exception:
                alert.notify("Image file not found.")


    def create_line_plot():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, (no_of_episodes * no_of_calls) + 1),
                 reward_hist,
                 marker='o',
                 color='b',
                 linestyle='-',
                 label='Objective')
        plt.title('Episode vs Objective Value')
        plt.xlabel('Episode')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.legend()
        plt.savefig("/Users/eugen/Downloads/line_plot.png")
        return


    def create_pair_plot():
        pplot_df = pd.DataFrame(iterated_parameters)
        sns.pairplot(pplot_df)
        current_time = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
        title = f"Pair Plot - Created on {current_time}"
        subtitle = f"Episodes: {no_of_episodes} - Calls made: {no_of_calls}"
        plt.suptitle(title, y=0.98, fontsize=14)
        plt.figtext(0.5, 0.95, subtitle, ha='center', fontsize=12, color='grey')
        plt.savefig("/Users/eugen/Downloads/pair_plot.png")
        return


    api_key = api_key_extractor.APIKeyExtractor()
    api_key = api_key.extract_api_key()

    tb = connect_to_telebot(api_key)

    if isinstance(tb, TeleBot):
        @tb.message_handler(commands=['start'])
        def main(message):
            if no_of_calls >= 10:
                alert = notify.Notify(tb, message_id=message)
                alert.initial_notify()

                n_calls = training_boundaries[0]
                param_space: list[tuple] = [(0.01, 0.5),  # Alpha (Learning rate)
                                            (0.8, 0.99),  # Gamma (Discount rate)
                                            (0.1, 1),  # Epsilon (Exploration rate)
                                            (1, 10),  # Decay
                                            (-5, 0),  # Bearish threshold
                                            (0, 5),  # Bullish threshold
                                            (0.01, 1)]  # EMA Difference limit

                result = gp_minimize(
                    lambda params: objective(params, alert, n_calls),
                    dimensions=param_space,
                    n_calls=n_calls,
                    random_state=42)
                summary_update(result, alert)
            else:
                logging.error("Too little no. of calls")


        tb.infinity_polling()
