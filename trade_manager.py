import logging
import constants
import numpy as np
from skopt import gp_minimize
import telebot
import requests
import api_key_extractor
from financial_instruments import macd
import matplotlib
import random
from training_helper import calculate_reward, graph_manager, q_table_manager, state_manager

matplotlib.use('Agg')


class TrainingAgent:
    _parameters: list[tuple] = [
        (0.50, 1.00),
        (0.80, 0.99),
        (0.00, 0.10),
        (1.00, 10.0),
        (-1.0, 1.00),
        (-1.0, 0.00),
    ]
    _number_of_episodes: int = 1
    _call_count: int = 0
    _number_of_calls: int = 10
    _number_of_omitted_rows: int = 810001
    _random_state: int = 42
    _iterated_values: dict = {}
    _reward_history: list = []

    def __init__(self):
        __macd = macd.MACD()
        self.dataset = __macd.calculate_macd()

        self.state_to_index = {state: state for state in range(len(constants.STATE_MAP))}

        self.n_of_omitted_rows = TrainingAgent._number_of_omitted_rows
        self.number_of_episodes = TrainingAgent._number_of_episodes
        self.iterated_values = TrainingAgent._iterated_values
        self.reward_history = TrainingAgent._reward_history

    @staticmethod
    def gaussian_process(tele_handler):
        return gp_minimize(
            func=lambda params: TrainingAgent.objective(params, tele_handler),
            dimensions=TrainingAgent._parameters,
            n_calls=TrainingAgent._number_of_calls,
            random_state=TrainingAgent._random_state,
            verbose=0)

    def build_graphs(self, result):
        training_agent = Trainer(result.x)
        pair_plot_handler = training_agent.get_pair_plot_handler
        pair_plot_handler.build_pair_plot(self.iterated_values)
        line_plot_handler = training_agent.get_line_plot_handler
        line_plot_handler.build_line_plot(self.reward_history,
                                          self.number_of_episodes,
                                          self._number_of_calls)

    @staticmethod
    def review_summary(tele_handler, result):
        graph_names: list[str] = ["pair_plot", "line_plot"]
        for path in graph_names:
            image_path = f"/Users/eugen/Downloads/{path}.png"
            try:
                with open(image_path, 'rb') as photo:
                    tele_handler.send_photo(photo, {path})
            except FileNotFoundError or Exception:
                tele_handler.send_message("Image file not found.")

        message_map: dict = {
            "alpha ": result.x[0],
            "gamma ": result.x[1],
            "epsilon ": result.x[2],
            "decay ": int(result.x[3]),
            "macd threshold ": result.x[4],
            "ema difference ": result.x[5],
        }

        for key, value in message_map.items():
            tele_handler.send_message(f"Best {key}:{value}")

    @classmethod
    def objective(cls, parameters, tele_handler):
        training_agent = Trainer(parameters)
        reward = training_agent.train()
        TrainingAgent._call_count = TrainingAgent._call_count + 1
        tele_handler.send_message(f"Call: {TrainingAgent._call_count}/{TrainingAgent._number_of_calls}")
        return -reward


class Trainer(TrainingAgent):
    def __init__(self, parameters):
        super().__init__()
        self.alpha, self.gamma, self.epsilon, self.decay, self.macd_threshold, self.ema_difference = parameters
        self.state_handler = state_manager.StateManager(self.macd_threshold,
                                                        self.ema_difference)
        self.reward_handler = calculate_reward.CalculateReward()

        self.q_table_handler = q_table_manager.QTableManager(self.alpha,
                                                             self.gamma)
        self.q_table = self.q_table_handler.create_q_table()

        self.pair_plot_handler = graph_manager.PairPlotManager(self.alpha,
                                                               self.gamma,
                                                               self.epsilon,
                                                               self.decay,
                                                               self.macd_threshold,
                                                               self.ema_difference)
        self.line_plot_handler = graph_manager.LinePlotManager

    def train(self):
        total_reward = 0
        for episode in range(self.number_of_episodes):
            episode_reward = 0
            for row_index in range(len(self.dataset) - self.n_of_omitted_rows):
                # Get state of current row
                current_row_content = self.dataset.iloc[row_index]
                current_state_index = self.state_handler.define_state(current_row_content)

                # Get state of next row
                next_row_content = self.dataset.iloc[row_index + 1]
                next_state_index = self.state_handler.define_state(next_row_content)

                # Choose course of action
                if random.uniform(0, 1) < self.epsilon:
                    action_index = random.choice(range(len(constants.AVAILABLE_ACTIONS)))
                else:
                    action_index = np.argmax(self.q_table[current_state_index])
                action = constants.AVAILABLE_ACTIONS[action_index]

                reward = self.reward_handler.calculate_reward(current_row_content,
                                                              next_row_content,
                                                              action)
                episode_reward += reward
                self.q_table = self.q_table_handler.update_q_table(self.q_table,
                                                                   current_state_index,
                                                                   next_state_index,
                                                                   action_index,
                                                                   reward)
            self.reward_history.append(episode_reward)
        self.iterated_values = self.pair_plot_handler.store_parameter_pair_plot(total_reward,
                                                                                self.iterated_values)
        return total_reward

    @property
    def get_pair_plot_handler(self):
        return self.pair_plot_handler

    @property
    def get_line_plot_handler(self):
        return self.line_plot_handler


class TeleBotManager:
    def __init__(self):
        self.api_key = self.get_api_key()

    @staticmethod
    def get_api_key():
        api_key = api_key_extractor.APIKeyExtractor()
        api_key = api_key.extract_api_key()
        return api_key

    def connect_tele_bot(self):
        """
        Attempts to connect to telegram bot.

        :return: Synchronous class for telegram bot.
        :rtype: Synchronous class
        """
        try:
            return telebot.TeleBot(self.api_key)
        except ValueError as e:
            logging.error("Invalid token {e]", e)
        except telebot.apihelper.ApiException as e:
            logging.error("Invalid token or connection error", e)
        except requests.exceptions.ConnectionError as e:
            logging.error("Network error:", e)


class Notifier(TeleBotManager):
    def __init__(self, t_bot, msgs):
        super().__init__()
        self.telebot = t_bot
        self.chat_id = msgs.chat.id

    def send_message(self, message):
        return self.telebot.send_message(self.chat_id, message)

    def send_photo(self, photo, message):
        return self.telebot.send_photo(self.chat_id, photo, caption=message)


if __name__ == "__main__":
    telebot = TeleBotManager().connect_tele_bot()


    @telebot.message_handler(commands=['train'])
    def train_model(message):
        tele_handler = Notifier(telebot, message)
        tele_handler.send_message("Training...")
        result = TrainingAgent().gaussian_process(tele_handler)
        TrainingAgent().build_graphs(result)
        TrainingAgent().review_summary(tele_handler, result)


    telebot.infinity_polling()
