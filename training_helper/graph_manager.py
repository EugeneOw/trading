import logging
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class PairPlotManager:
    def __init__(self, alpha, gamma, epsilon, decay, macd_threshold, ema_difference):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.macd_threshold = macd_threshold
        self.ema_difference = ema_difference
        self.episode_reward = None

    def store_parameter_pair_plot(self, total_reward, iterated_values):
        try:
            for key, value in {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'decay': int(self.decay),
                'total_reward': float(total_reward)
            }.items():

                if key in iterated_values:
                    iterated_values[key].append(value)
                else:
                    iterated_values[key] = [value]
            return iterated_values
        except AttributeError as e:
            logging.error(f"AttributeError: {e} - Ensure self.iterated_parameters is initialized as a dictionary.")
        except TypeError as e:
            logging.error(f"TypeError: {e} - Ensure self.iterated_parameters[key] is a list before calling append.")
        except Exception as e:
            logging.error(f"Unexpected error occurred while updating iterated_parameters for {key}: {e}")

    @staticmethod
    def build_pair_plot(all_iterated_values):
        pair_plot_diagram = pd.DataFrame(all_iterated_values)
        sns.pairplot(pair_plot_diagram)
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        title = f"Pair Plot - Created on {current_time}"
        plt.suptitle(title, y=0.98, fontsize=14)
        plt.figtext(0.5, 0.95,  "", ha='center', fontsize=12, color='grey')
        plt.savefig("/Users/eugen/Downloads/pair_plot.png")


class LinePlotManager:
    @staticmethod
    def build_line_plot(all_rewards, no_of_episodes, no_of_calls):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, (no_of_episodes * no_of_calls) + 1),
                 all_rewards,
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
