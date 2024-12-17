import os
import math
import random
import telebot
import matplotlib
import numpy as np
from skopt import Optimizer
from constants import constants as c
from collections import namedtuple
from live_data import live_fx_data
from financial_instruments import macd
from telebot_manager import telebot_manager
from database_manager import database_manager
from concurrent.futures import ThreadPoolExecutor
from training_helper import reward_manager, graph_manager, q_table_manager, state_manager

matplotlib.use('Agg')


class TrainingAgent:
    calls: int = 1
    episodes: int = 1
    random_state: int = 42
    omit_rows: int = 810001  # Minimum: 1
    
    reward_hist: list[float] = []  # Stores rewards to use in line graph
    iter_values: dict[str: list[float]] = {}  # Stores values that were tested to achieve most optimize reward.
    
    parameters: list[tuple] = c.PARAM_TRAINING

    def __init__(self):
        macd_handler = macd.MACD()
        self.dataset = macd_handler.calculate_macd()
        
        self.state_to_index = self.create_state_map
        
        self.omit_rows = TrainingAgent.omit_rows
        self.episodes = TrainingAgent.episodes
        self.iter_values = TrainingAgent.iter_values
        self.reward_history = TrainingAgent.reward_hist

    @property
    def create_state_map(self):
        return {state: state for state in range(len(c.STATE_MAP))}

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
        optimizer = Optimizer(dimensions=TrainingAgent.parameters, random_state=TrainingAgent.random_state)
        
        for call in range(TrainingAgent.calls):
            params = optimizer.ask()  # 'TrainingAgent.parameters'
            reward = TrainingAgent.objective(params)
            optimizer.tell(params, reward)
            tele_handler.send_message(f"Call iteration: {call + 1}/{TrainingAgent.calls}\n")
            
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
        pair_plot_handler.build_pair_plot(self.iter_values)
        
        line_plot_handler = training_agent.get_line_plot_handler
        line_plot_handler.build_line_plot(self.reward_history, self.episodes, self.calls)

    @staticmethod
    def review_summary(tele_handler, best_params):
        """
        Sends a summary of the optimization result and graphs via Telegram

        :param tele_handler: An instance of a handler to send messages through Telegram
        :param best_params: The best parameters found during optimization
        :return: None
        """
        best_params_message = "\n".join(f"{param} : {round(value, 4)}" for param, value in zip(c.PARAM_NAME, best_params))
        tele_handler.send_message("Optimization complete!")
        tele_handler.send_message(best_params_message)
        
        db_file = os.path.abspath(c.PATH_DB)
        file_path_manager = database_manager.FilePathManager(db_file)
        file_path = file_path_manager.fetch_file_path(1)

        for graph in c.AVAIL_GRAPHS:
            image_path = f"{file_path}{graph}.png"
            try:
                with open(image_path, 'rb') as photo:
                    tele_handler.send_photo(photo, {graph})
            except FileNotFoundError or Exception:
                tele_handler.send_message("Image file not found.")


class Trainer(TrainingAgent):
    def __init__(self, parameters):
        super().__init__()
        [self.alpha, self.gamma, self.decay, self.macd_threshold, self.max_grad, self.scale_fac, self.grad, self.mid] = parameters

        self.next_instr: int = 0  # Contains instrument used to determine next row's state (Bull/Bearish)
        self.current_instr: int = 0

        self.episode: int = 0
        self.row_idx: int = 0
        self.length_of_dataset: int = 0

        self.curr_row: list = []
        self.curr_state_idx: int = 0
        self.curr_instr: int = 0

        self.prev_row: list = []
        self.prev_state_idx = None  # Set as None, since 0 is a possible choice
        self.prev_instr: int = 0

        self.next_row: list = []
        self.next_state_idx: int = 0
        self.next_instr: int = 0

        self.total_decay: float = 0  # Not same as 'decay' which is a constant used to calculate 'total_decay'

        self.state_handler = state_manager.StateManager(self.macd_threshold, self.decay)
        self.instr_weight = self.state_handler.create_weights()

        self.reward_handler = reward_manager.CalculateReward(self.max_grad, self.scale_fac, self.grad, self.mid)

        self.q_table_handler = q_table_manager.QTableManager(self.alpha, self.gamma)
        self.q_table = self.q_table_handler.create_q_table()

        self.pair_plot_handler = graph_manager.PairPlotManager(self.alpha, self.gamma, self.decay, self.macd_threshold)
        self.line_plot_handler = graph_manager.LinePlotManager()

    def train(self):
        """
        Trains the Q-learning agent over a defined number of episodes (episodes) using the given parameters
        and dataset.

        This method iterates through the dataset row by row for each episode. At each step, it will;
            - Determine the current and next row's state.
            - Selects an action based on a decay-greedy policy.
            - Calculate the reward for the chosen action.
            - Updates the Q-table using the calculated reward(reward)
            - Tracks the cumulative reward for the episode

        After completing all episodes, it stores the parameters pair for plotting.

        :return: The total reward accumulated over all episodes.
        """
        total_reward = 0
        for episode in range(self.episodes):
            eps_reward = 0
            self.episode = episode
            self.length_of_dataset = len(self.dataset) - self.omit_rows
            for row_idx in range(self.length_of_dataset):
                self.row_idx = row_idx

                self.get_curr_state()
                self.get_next_state()

                self.prev_row = self.next_row
                self.prev_state_idx = self.next_state_idx

                act_idx = self.course_of_action()

                # Calculates reward based on chosen action
                reward, self.instr_weight = self.reward_handler.calc_reward(self.curr_row, self.next_row, act_idx, self.row_idx, self.instr_weight,
                                                                            self.current_instr, episode)
                eps_reward += reward

                # Updates q-table
                self.q_table = self.q_table_handler.update_q_table(self.q_table, self.curr_state_idx, self.next_state_idx, act_idx, reward)

            self.reward_hist.append(eps_reward)
            total_reward += eps_reward

        self.iter_values = self.pair_plot_handler.store_parameter_pair_plot(total_reward, self.iter_values)
        return total_reward, self.q_table

    def get_curr_state(self):
        if len(self.prev_row) != 0 and self.prev_state_idx is not None:
            # Since previous row has already been identified, we don't have to look through data set again.
            self.curr_row, self.curr_state_idx = self.prev_row, self.prev_state_idx
            self.current_instr = self.next_instr

        else:
            # Happens only initially when 'prev_row' and 'prev_state_idx' = None
            self.curr_row = self.dataset.iloc[self.row_idx]
            self.curr_state_idx, self.current_instr = self.state_handler.define_state(self.curr_row, self.instr_weight, self.decay)

    def get_next_state(self):
        self.next_row = self.dataset.iloc[self.row_idx + 1]
        self.next_state_idx, self.next_instr = self.state_handler.define_state(self.next_row, self.instr_weight, self.decay)

    def course_of_action(self):
        """
        Selects next course of action based on a decaying effect that is similar to an exp ** -x
        :return: c.AVAIL_ACTIONS[action_idx]:
        """
        self.total_decay = self.calc_total_decay
        if random.uniform(0, 1) < self.decay:
            length_actions = range(len(c.AVAIL_ACTIONS))
            return random.choice(length_actions)
        else:
            if random.uniform(0, 1) < 0.1:  # 10% to select wrong value
                return np.argmin(self.q_table[self.curr_state_idx])
            else:
                return np.argmax(self.q_table[self.curr_state_idx])

    @property
    def calc_total_decay(self):
        constant = self.decay
        current_iteration = (self.row_idx + 1) * (self.episode + 1)
        total_iterations = self.length_of_dataset * self.episodes
        return float(math.exp(-constant * (current_iteration / total_iterations)))

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

        # Resets '_reward_history' to prevent re-running from wrongly
        # appending '_reward_history' to previous attempts.
        TrainingAgent.reward_history = []

        # Trains model to get most optimized parameters.
        training_agent = TrainingAgent()
        result, best_params = training_agent.gaussian_process(tele_handler)

        # Builds graph and sends telegram message to inform completion.
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

        reward, q_table = Trainer(c.OPTIMIZE_PARAM).train()  # reward needed only for training

        tele_handler.send_message("Training done.")
        tele_handler.send_table(q_table)
        tele_handler.send_message(f"Total reward: {reward}")

        db_file = os.path.abspath(c.Q_TABLE_DB)
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
        live_fx_handler = live_fx_data.LiveFX()
        executor = ThreadPoolExecutor(max_workers=2)
        stream_data = executor.submit(live_fx_handler.get_stream())


    telebot.infinity_polling()
