import os
import math
import pytz
import random
import telebot
import matplotlib
import numpy as np
import pandas as pd
from skopt import Optimizer
from datetime import datetime
from collections import namedtuple
from live_data import live_fx_data
from constants import constants as c
import dataset_manager.dataset_manager
from news_manager import news_manager
from dataset_manager import dataset_manager
from telebot_manager import telebot_manager
from database_manager import database_manager
from training_helper import reward_manager, graph_manager, q_table_manager, state_manager

matplotlib.use('Agg')


class TrainingAgent:
    reward_hist: list[float] = []  # Stores rewards to use in line graph
    iter_values: dict[str: list[float]] = {}  # Stores values that were tested to achieve most optimize reward.

    calls: int = c.CALLS
    episodes: int = c.EPISODES
    random_state: int = c.RANDOM_STATE
    omit_rows: int = c.OMIT_ROWS
    parameters: list[tuple] = c.PARAM_TRAINING

    def __init__(self):
        self.state_to_index = self.create_state_map
        self.dataset = pd.read_csv(self.get_dataset)

        self.omit_rows = TrainingAgent.omit_rows
        self.episodes = TrainingAgent.episodes
        self.iter_values = TrainingAgent.iter_values
        self.reward_history = TrainingAgent.reward_hist

        print("Initialize")

    @property
    def get_dataset(self):
        db_file = os.path.abspath(c.PATH_DB)
        database_manager.DBManager(db_file)
        file_path_manager = database_manager.FilePathManager(db_file)
        return file_path_manager.fetch_file_path(4)

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
            tele_handler.send_message(f"<b>Call iteration:</b> <i>{call + 1}/{TrainingAgent.calls}</i>\n")

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
        best_params_message = "\n".join(f"<b>{param}:</b> <i><code>{value}</code></i>" for param, value in zip(c.PARAM_NAME, best_params))
        tele_handler.send_message(best_params_message)

        db_file = os.path.abspath(c.PATH_DB)
        file_path_manager = database_manager.FilePathManager(db_file)
        file_path = file_path_manager.fetch_file_path(1)

        singapore_timezone = pytz.timezone("Asia/Singapore")

        for graph in c.AVAIL_GRAPHS:
            image_path = f"{file_path}{graph}.png"
            try:
                singapore_time = datetime.now(singapore_timezone)
                formatted_now = singapore_time.strftime("%d-%m-%Y %H:%M:%S")
                with open(image_path, 'rb') as photo:
                    message = f"<b>Graph Type:</b> <i>{graph}</i>\n<b>Time:</b> <i>{str(formatted_now)}</i>"
                    tele_handler.send_photo(photo, {message})
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

    @telebot.message_handler(commands=['build'])
    def build(message):
        """
        Builds ('dataset_manager') and update ('forex_data') with a new dataset that
        we can process with 'financial_instruments'
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("<b>Building new dataset</b> - <i>Please await completion message.</i>")
        dataset_manager.DatasetManager()
        tele_handler.send_message("<i>New dataset has finished building.</i>")


    @telebot.message_handler(commands=['optimize'])
    def optimize(message):
        """
        Handles the '/optimize' command to initiate the training process for the model.
        Gets the best parameters by train data
        :param message: Message object containing details of the '/optimize' command
        :return: None
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("<b>Initiating optimizing</b> - <i>Please await completion message.</i>")

        # Resets '_reward_history' to prevent re-running from wrongly
        # appending '_reward_history' to previous attempts.
        TrainingAgent.reward_history = []

        # Trains model to get most optimized parameters.
        training_agent = TrainingAgent()
        result, best_params = training_agent.gaussian_process(tele_handler)

        # Builds graph and sends telegram message to inform completion.
        training_agent.build_graphs(result)
        training_agent.review_summary(tele_handler, best_params)


    @telebot.message_handler(commands=['train'])
    def train_model(message):
        """
        Utilizes optimize parameters to get optimize q-table
        :param message: Message object containing details of the '/train' command
        :return: None
        """
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message("<b>Initiating training</b> - <i>Please await completion message.</i>")

        reward, q_table = Trainer(c.OPTIMIZE_PARAM).train()  # reward needed only for training

        tele_handler.send_message("<b>Training done.</b>")
        tele_handler.send_table(q_table)
        tele_handler.send_message(f"<b>Total reward:</b> <i>{reward}</i>")

        db_file = os.path.abspath(c.Q_TABLE_DB)
        database_manager.DBManager(db_file)
        q_table_db_manager = database_manager.QTableManager(db_file)
        q_table_db_manager.q_table_operation(q_table)
        tele_handler.send_message("<i>Updated q-table has been stored.</i>")


    @telebot.message_handler(commands=['test'])
    def test_model(message):
        """
        Get live data from Oanda API
        :param message: Message object containing details of the '/test' command
        :return: None
        """
        live_fx_handler = live_fx_data.LiveFX()
        live_fx_handler.get_stream()
        #executor = ThreadPoolExecutor(max_workers=2)
        #stream_data = executor.submit(live_fx_handler.get_stream())

    @telebot.message_handler(commands=['news'])
    def retrieve_news(message):
        tele_handler = telebot_manager.Notifier(telebot, message)
        tele_handler.send_message(f"<b>Retrieving news</b> - <i>Please await completion message.</i>")

        news_handler = news_manager.NewsManager()
        news_handler.setUp()
        content = news_handler.start_web_scrap()

        for url, message in content.items():
            tele_handler.send_message(f"\n<b>{message[0].upper()}:</b> \n{message[1]}\n{url}")
        tele_handler.send_message(f"<i>{c.ARTICLES} articles has been retrieved.</i>")

    telebot.infinity_polling()
