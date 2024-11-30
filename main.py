import ema
import random

class Main:
    def __init__(self):
        self.file_path:str = '/Users/eugen/Downloads/dataset.csv'
        self.df = ema.EMA(self.file_path)

        self.reward_goal = 1
        self.reward_step = -1
        self.actions = ['Long', 'Short']
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.episodes = 1000


class Sub(Main):
    # Function to choose action based on epsilon-greedy strategy
    def action(self):
        if random.uniform(0, 1)<self.epsilon:



if __name__ == "__main__":
    Sub().env()