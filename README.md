# Trading Agent with Reinforcement Learning and Optimization

This repository contains the code for a trading agent that leverages reinforcement learning and optimization techniques. The agent makes trading decisions (buy, sell, or hold) based on a combination of technical indicators like **MACD** (Moving Average Convergence Divergence), **EMA** (Exponential Moving Average), and other market data.

---
## Overview

The system is designed to train a reinforcement learning agent to perform trades using historical forex data. The agent learns from the market data using indicators such as **MACD** and **EMA**, and optimizes its trading strategy based on reward maximization.

---
### Trading Strategy:
The agent's trading strategy is based on a range of instruments such as:
- **MACD** (Moving Average Convergence Divergence)
- **EMA** (Exponential Moving Average)

These indicators are used to determine market trends (Bullish, Bearish, or Neutral) and help make decisions on whether to **Buy**, **Sell**, or **Hold** based on market conditions.

---
### Requirements

- Python 3.6 or higher
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-optimize
  - telebot
  - requests
  - prettytable


---
### File Structure

	•	macd.py: Contains the MACD class that computes the MACD values and integrates EMA calculations.
	•	ema.py: Defines the EMA class that calculates the Exponential Moving Averages for periods of 12 and 24.
	•	main.py: Main script that sets up the training loop, optimizes parameters, integrates the Telegram bot, and generates plots.
	•	notify.py: Contains the Notify class for sending alerts and notifications via Telegram.
	•	api_key_extractor.py: Extracts the Telegram bot API key.
	•	dataset.csv: Sample dataset containing forex data (you must replace it with your own data file).
---
### How It Works

1. **Data Loading & Preprocessing:**

	MACD and EMA values are computed using forex data loaded from a CSV file (dataset.csv). 
	The MACD class is used to calculate the MACD and the EMA class computes EMA for periods 12 and 24. This data is then used for the Q-learning agent’s decision-making process.


2.  **Q-Learning Agent :**

	Sub class inherits from Main and defines the states, actions, and the Q-table for the agent. 
	The agent’s training involves selecting actions based on the states defined by the MACD and EMA values and updating the Q-table using the Q-learning formula. 
	Training occurs in episodes, and after each episode, the total reward is calculated and stored.


3. **Parameter Optimization:**
	
	gp_minimize function from skopt optimizes the Q-learning parameters (alpha, gamma, epsilon, and decay) to maximize the agent’s performance over training episodes.


4. **Telegram Notifications & Alerts:**

	Notify class sends real-time updates via Telegram, notifying the user of training progress, parameter updates, and rewards. Plots of the training performance (line plots and pair plots) are generated and sent to the user.


5. **Plotting & Visualization:**

	Line plots showing the relationship between the episodes and the objective value are created.
	Pair plots visualize the relationship between different parameters (alpha, gamma, epsilon, decay).

6. **Running the Bot:**

    Telegram bot will handle the start command and begin the training process, notifying the user of progress and results. The bot will send the final optimized parameters and training plots as images.
---
### Usage

1.	Set Up API Key:
	
		Exract your Telegram Bot API key (from BotFather) and place it in the api_key_extractor.py file.


2. **Run the Script:**

	Execute the script using:
	```python main.py```


3. **Getting bot started:**

	Once the bot starts, send ```/start``` to the bot to begin the training process.


4.  **Training Process:**

	Training process will run for the defined number of episodes and calls, optimizing the Q-learning parameters. You will receive updates in your Telegram with details about the training process, including parameters and rewards.


5. **Viewing Results:**

	After the training process, the bot will send you a summary message containing the best parameters found during optimization and share visual plots (line plot and pair plot).

---
### Example output: 

```
Best alpha: 0.12
Best gamma: 0.95
Best epsilon: 0.85
Best decay: 4

[Line plot image]
[Pair plot image]
```
---
#### Customization

1. **File Path:** 
	
	Modify the file_path variable in the Main class to point to your own forex data CSV file.
   

2. **States and Actions:** 
   
   You can adjust the states and actions in the Sub class to fit other types of market data or strategies.


3. **Training Boundaries:** 
   
   Adjust the training_boundaries and no_of_calls to control the number of training episodes and optimization calls.

