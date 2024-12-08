
import time
import random
import telebot
import websocket
import matplotlib
import logging
import numpy as np
import thread
import yfinance as yf
from telebot_manager import telebot_manager
from skopt import Optimizer
from constants import constants
from collections import namedtuple
from financial_instruments import macd
from training_helper import calculate_reward, graph_manager, q_table_manager, state_manager

matplotlib.use('Agg')


class TrainingAgent:
    _parameters: list[tuple] = [
        (0.00, 1.00),
        (0.70, 1.00),
        (-1.0, 0.10),
        (1.00, 10.0),
        (-1.0,    1.00),
        (-1.0, 0.00),
    ]
    _number_of_episodes: int = 5
    _call_count: int = 0
    _number_of_calls: int = 30
    _number_of_omitted_rows: int = 700000
    _random_state: int = 42
    _iterated_values: dict = {}
    _reward_history: list = []
    current_status: list = []
    current_episode: int = 0

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
        """
        Executes a Gaussian Process optimization to tune model parameters. This method optimizes a set of
        parameters using a Gaussian Process optimizer.

        :param tele_handler: An instance of a handler to send messages through Telebot
        :return: A tuple containing:
            - result:
                - x: The best parameters found.
                - fun: The best (minimum) value
                - xi: List of all parameters set evaluated.
                - yi: List of corresponding objective function values.
            - best_params: The best parameters as a separate list.
        """
        optimizer = Optimizer(dimensions=TrainingAgent._parameters,
                              random_state=TrainingAgent._random_state)
        for call in range(TrainingAgent._number_of_calls):
            params = optimizer.ask()
            reward = TrainingAgent.objective(params)
            optimizer.tell(params, reward)
            formatted_params = [
                f"Best {name}   {value:.3f}" for name, value in zip(constants.PARAMETERS_NAME, params)
            ]
            formatted_params = "\n".join(formatted_params)
            tele_handler.send_message(
                f"Iteration   {call + 1}/{TrainingAgent._number_of_calls}\n"
                f"{formatted_params}\n"
                f"Reward   {-reward}\n"
            )

        best_idx = np.argmin(optimizer.yi)
        best_params = optimizer.Xi[best_idx]
        best_value = optimizer.yi[best_idx]
        Result = namedtuple("Result", ["x", "fun", "xi", "yi"])
        result = Result(x=best_params, fun=best_value, xi=optimizer.Xi, yi=optimizer.yi)

        return result, best_params

    @classmethod
    def objective(cls, parameters):
        """
        Evaluates the objective function for a given set of parameters.

        :param parameters: A list of parameters to evaluate
        :type parameters: list

        :return: The negative reward as a measure of the objective function's performance
        :rtype: float
        """
        training_agent = Trainer(parameters)
        reward = training_agent.train()
        return -reward

    def build_graphs(self, result):
        """
        Builds the graph required to visualize the data

        :param result: A named tuple containing the optimization result.
        :return: None
        """
        training_agent = Trainer(result.x)
        pair_plot_handler = training_agent.get_pair_plot_handler
        pair_plot_handler.build_pair_plot(self.iterated_values)
        line_plot_handler = training_agent.get_line_plot_handler
        line_plot_handler.build_line_plot(self.reward_history,
                                          self.number_of_episodes,
                                          self._number_of_calls)

    @staticmethod
    def review_summary(tele_handler, best_params):
        """
        Sends a summary of the optimization result and graphs via Telegram

        :param tele_handler: An instance of a handler to send messages through Telegram
        :param best_params: The best parameters found during optimization
        :return: None
        """
        best_params_message = "\n".join(
            f"Best {name}:{value}" for name, value in zip(constants.PARAMETERS_NAME, best_params)
        )
        tele_handler.send_message(f"Optimization complete!\n{best_params_message}")

        graph_names: list[str] = ["pair_plot", "line_plot"]
        for path in graph_names:
            image_path = f"/Users/eugen/Downloads/{path}.png"
            try:
                with open(image_path, 'rb') as photo:
                    tele_handler.send_photo(photo, {path})
            except FileNotFoundError or Exception:
                tele_handler.send_message("Image file not found.")


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
        """
        Trains the Q-learning agent over a defined number of episodes (number_of_episodes) using the given parameters
        and dataset.

        This method iterates through the dataset row by row for each episode. At each step, it will;
            - Determine the current and next row's state.
            - Selects an action based on an epsilon-greedy policy.
            - Calculate the reward for the chosen action.
            - Updates the Q-table using the calculated reward(reward)
            - Tracks the cumulative reward for the episode

        After completing all episodes, it stores the parameters pair for plotting.

        :return: The total reward accumulated over all episodes.
        """
        total_reward = 0
        start_time = time.time()
        for episode in range(self.number_of_episodes):
            TrainingAgent.current_episode = episode
            episode_reward = 0
            previous_row_content = None
            previous_state_index = None
            for row_index in range(len(self.dataset) - self.n_of_omitted_rows):
                # Get state of current row
                if previous_row_content is not None and previous_state_index is not None:
                    current_row_content, current_state_index = previous_row_content, previous_state_index
                else:
                    current_row_content = self.dataset.iloc[row_index]
                    current_state_index = self.state_handler.define_state(current_row_content)

                # Get state of next row
                next_row_content = self.dataset.iloc[row_index + 1]
                next_state_index = self.state_handler.define_state(next_row_content)
                previous_row_content, previous_state_index = next_row_content, next_state_index

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
            self.reward_history.append(episode_reward*-1)
            total_reward += episode_reward
        self.iterated_values = self.pair_plot_handler.store_parameter_pair_plot(total_reward,
                                                                                self.iterated_values)
        logging.info(f"Time taken per episode {time.time()-start_time}")
        return total_reward

    @property
    def get_pair_plot_handler(self):
        """
        Provides access to the pair plot handler instance.

        :return: The pair plot handler object associate with this instance.
        """
        return self.pair_plot_handler

    @property
    def get_line_plot_handler(self):
        """
        Provides access to the line plot handler instance.

        :return: The line plot handler object associate with this instance.
        """
        return self.line_plot_handler


if __name__ == "__main__":
    tele_bot = telebot_manager.TeleBotManager()
    telebot = tele_bot.connect_tele_bot()

    @telebot.message_handler(commands=['train'])
    def train_model(message):
        """
        Handles the '/train' command to initiate the training process for the model.

        :param message: Message object containing details of the '/train' command
        :return: None
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("Initiating training")
        result, best_params = TrainingAgent().gaussian_process(tele_handler)
        TrainingAgent().build_graphs(result)
        TrainingAgent().review_summary(tele_handler,  best_params)

    @telebot.message_handler(commands=['test'])
    def test_model(message):
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("Initiating testing")
        TestingAgent().display()
    telebot.infinity_polling()
