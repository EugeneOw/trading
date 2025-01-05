import os
import logging
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from constants import constants as c
from database_manager import database_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class GraphManager:
    def __init__(self):
        self.db_file = os.path.abspath(c.PATH_DB)
        database_manager.DBManager(self.db_file)
        self.file_path_manager = database_manager.FilePathManager(self.db_file)


class PairPlotManager(GraphManager):
    def __init__(self, alpha, gamma, epsilon, macd_threshold):
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.macd_threshold: float = macd_threshold

        self.episode_reward = None

    def store_parameter_pair_plot(self, total_reward, iter_values):
        """
        Creates (if !exists) and updates a dictionary(iterated_values) that contains all the parameters used in the
        training and the final reward (total_reward).

        :param total_reward: Contains the final reward value calculated and cumulated at the end of every call.
        :type total_reward: float

        :param iter_values: Contains all parameters and final reward.
        :type iter_values: dict

        :return: iterated_values: Returns the dictionary back to training agent to update for next cycle.
        :rtype: iterated_values: dict
        """
        try:
            for key, value in {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'macd': self.macd_threshold,
                'total_reward': float(total_reward)
            }.items():

                if key in iter_values:
                    iter_values[key].append(value)
                else:
                    iter_values[key] = [value]
            return iter_values

        except AttributeError as e:
            logging.error(f"AttributeError: {e} - Ensure self.iterated_parameters is initialized as a dictionary.")
        except TypeError as e:
            logging.error(f"TypeError: {e} - Ensure self.iterated_parameters[key] is a list before calling append.")

    def build_pair_plot(self, all_iterated_values):
        """
        Builds and saves a pair-plot diagram that shows how different parameters affect other parameters and more
        importantly the final objective (total_reward).

        :param all_iterated_values: Contains all parameters (of that call) and its final reward (total_reward).
        :type all_iterated_values: dict

        :return: None
        """
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        title = f"Pair Plot - Created on {current_time}"

        pair_plot_diagram = pd.DataFrame(all_iterated_values)
        sns.pairplot(pair_plot_diagram)

        plt.suptitle(title, y=0.98, fontsize=14)
        plt.figtext(0.5, 0.95, "", ha='center', fontsize=12, color='grey')

        file_path = self.file_path_manager.fetch_file_path(1)

        plt.savefig(f"{file_path}/pair plot.png")


class LinePlotManager(GraphManager):
    def __init__(self):
        super().__init__()

    def build_line_plot(self, all_rewards, no_of_episodes, no_of_calls):
        """
        Builds and saves a line plot that shows how the reward (per episode) changes as training goes on.

        :param all_rewards: Contains all the rewards
        :type all_rewards: list[float]

        :param no_of_episodes: Contains the number of episodes
        :type no_of_episodes: int

        :param no_of_calls: Contains the number of calls made
        :type no_of_calls: int

        :return: None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, (no_of_episodes * no_of_calls) + 1),
                 all_rewards,
                 marker='o', color='b', linestyle='-', label='Reward')

        plt.title('Objective')
        plt.xlabel('Episode')
        plt.ylabel('Objective Reward')
        plt.grid(True)
        plt.legend()

        file_path = self.file_path_manager.fetch_file_path(1)
        plt.savefig(f"{file_path}/line plot.png")
