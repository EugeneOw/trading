import os
import random
import telebot
import matplotlib
import numpy as np
from skopt import Optimizer
from constants import constants
from collections import namedtuple
from live_data import live_fx_data
from financial_instruments import macd
from telebot_manager import telebot_manager
from database_manager import database_manager
from training_helper import reward_manager, graph_manager, q_table_manager, state_manager
import logging

matplotlib.use('Agg')


class TrainingAgent:
    _reward_history: list = []
    _iterated_values: dict = {}

    _call_count: int = 0
    _random_state: int = 42
    _number_of_calls: int = 1
    _number_of_episodes: int = 5
    _number_of_omitted_rows: int = 1  # Minimum: 1

    _parameters: list[tuple] = constants.PARAMETERS_TRAINING

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
        :param: tele_handler: Handles the telegram send_message method.
        :param: _optimize_episodes: None
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
            tele_handler.send_message(
                f"Call iteration: {call + 1}/{TrainingAgent._number_of_calls}\n"
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
        reward, q_table = training_agent.train()  # q_table needed only for optimize
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
            f"Best {name} : {value}" for name, value in zip(constants.PARAMETERS_NAME, best_params)
        )
        tele_handler.send_message(f"Optimization complete!\n{best_params_message}")

        graph_names: list[str] = ["pair plot", "line plot"]
        db_file = os.path.abspath(constants.PATH_DB)
        file_path_manager = database_manager.FilePathManager(db_file)
        file_path = file_path_manager.fetch_file_path(1)

        for graph in graph_names:
            image_path = f"{file_path}{graph}.png"
            try:
                with open(image_path, 'rb') as photo:
                    tele_handler.send_photo(photo, {graph})
            except FileNotFoundError or Exception:
                tele_handler.send_message("Image file not found.")


class Trainer(TrainingAgent):
    def __init__(self, parameters):
        super().__init__()
        [self.alpha,
         self.gamma,
         self.epsilon,
         self.decay,
         self.macd_threshold,
         self.ema_difference,
         self.max_gradient,
         self.scaling_factor,
         self.gradient,
         self.midpoint] = parameters

        self.next_instrument = ""
        self.current_instrument = 0
        self.row_index = 0

        self.state_handler = state_manager.StateManager(self.macd_threshold, self.ema_difference, self.epsilon)
        self.instrument_weight = self.state_handler.create_weights()
        self.reward_handler = reward_manager.CalculateReward(self.max_gradient, self.scaling_factor, self.gradient,
                                                             self.midpoint)

        self.q_table_handler = q_table_manager.QTableManager(self.alpha, self.gamma)
        self.q_table = self.q_table_handler.create_q_table()

        self.pair_plot_handler = graph_manager.PairPlotManager(self.alpha, self.gamma, self.epsilon, self.decay,
                                                               self.macd_threshold, self.ema_difference)
        self.line_plot_handler = graph_manager.LinePlotManager()

    def course_of_action(self, curr_state_idx):
        """

        :param curr_state_idx:
        :return: constants.AVAILABLE_ACTIONS[action_idx]:
        """
        if random.uniform(0, 1) < self.epsilon:
            if random.uniform(0, 1) < 0.1:  # 10% to select wrong value
                return np.argmin(self.q_table[curr_state_idx])
            else:
                length_actions = range(len(constants.AVAILABLE_ACTIONS))
                return random.choice(length_actions)
        else:
            return np.argmax(self.q_table[curr_state_idx])

    def get_curr_state(self, previous_row_content, previous_state_index):
        if previous_row_content is not None and previous_state_index is not None:

            # Since previous row has already been identified, we don't have to look through data set again.
            current_row_content, current_state_index = previous_row_content, previous_state_index
            self.current_instrument = self.next_instrument
        else:
            current_row_content = self.dataset.iloc[self.row_index]
            current_state_index, self.current_instrument = self.state_handler.define_state(current_row_content, self.instrument_weight)
        return current_row_content, current_state_index

    def get_next_state(self):
        next_row_content = self.dataset.iloc[self.row_index + 1]
        next_state_index, self.next_instrument = self.state_handler.define_state(next_row_content, self.instrument_weight)
        return next_row_content, next_state_index

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
        for episode in range(self.number_of_episodes):
            episode_reward = 0
            previous_row_content = None
            previous_state_index = None

            for row_index in range(len(self.dataset) - self.n_of_omitted_rows):
                self.row_index = row_index
                # Get state of current row
                current_row_content, current_state_index = self.get_curr_state(previous_row_content, previous_state_index)

                # Get state of next row
                next_row_content, next_state_index = self.get_next_state()
                previous_row_content, previous_state_index = next_row_content, next_state_index

                # Choose course of action
                action_index = self.course_of_action(current_state_index)

                # Calculates reward based on chosen action
                reward, updated_instrument_weight = self.reward_handler.calculate_reward(current_row_content, next_row_content, action_index, row_index,
                                                                                         self.instrument_weight, self.current_instrument, episode)

                self.instrument_weight = updated_instrument_weight
                episode_reward += reward

                # Updates q-table
                self.q_table = self.q_table_handler.update_q_table(self.q_table, current_state_index, next_state_index, action_index, reward)

            self.reward_history.append(episode_reward)
            total_reward += episode_reward
        self.iterated_values = self.pair_plot_handler.store_parameter_pair_plot(total_reward, self.iterated_values)

        return total_reward, self.q_table

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


    @telebot.message_handler(commands=['optimize'])
    def optimize(message):
        """
        Handles the '/optimize' command to initiate the training process for the model.
        Gets the best parameters by train data
        :param message: Message object containing details of the '/optimize' command
        :return: None
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("Initiating optimizing.")

        # Resets '_reward_history' to prevent re-running from wrongly updating _reward_history from previous attempts.
        TrainingAgent._reward_history = []

        result, best_params = TrainingAgent().gaussian_process(tele_handler)
        TrainingAgent().build_graphs(result)
        TrainingAgent().review_summary(tele_handler, best_params)


    @telebot.message_handler(commands=['train'])
    def train_model(message):
        """
        Utilizes optimize parameters to get optimize q-table
        :param message: Message object containing details of the '/train' command
        :return: None
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("Initiating training - Please await completion message.")

        reward, q_table = Trainer(constants.OPTIMIZE_PARAMETERS).train()  # reward needed only for training

        tele_handler.send_message("Training done.")
        tele_handler.send_table(q_table)
        tele_handler.send_message(f"Total reward: {reward}")

        db_file = os.path.abspath(constants.Q_TABLE_DB)
        database_manager.DBManager(db_file)
        q_table_db_manager = database_manager.QTableManager(db_file)
        q_table_db_manager.q_table_operation(q_table)
        tele_handler.send_message("Updated q-table has been stored.")


    @telebot.message_handler(commands=['test'])
    def test_model(message):
        """
        Get live data from Oanda API
        :param message: Message object containing details of the '/test' command
        :return: None
        """
        live_fx_data.LiveFX().get_stream()


    telebot.infinity_polling()
