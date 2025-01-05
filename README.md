# Trading Agent with Reinforcement Learning and Optimization
## Overview
This repository contains the code for an agent that can:

 1.	```\build``` (Re)build a dataset with real-time financial news and indicators.
 2.	```\optimize``` Optimize training parameters (Bayesian optimization).
 3.	```\train``` Train using reinforcement learning (Q-learning).
 4.	```\test``` Test on real-time data (Oanda API).
 5.	```\news``` Gather real-time financial news (Selenium web-scrapping + Gemini API).

The agent is operated and managed through the use of a Telegram Bot, which also provides real-time status updates and results. This was chosen purely
for the ease of user experience and flexibility.

---
## Installation
### Pre-requisites
  - Python 3.7 or higher
### Required libraries
    - pytz
    - numpy
    - skopt
    - pandas
    - seaborn 
    - telebot
    - requests
    - matplotlib
    - prettytable
    - collections
    - scikit-optimize

### Steps to Install
1. Clone the repository:
    ```git clone https://github.com/EugeneOw/trading.git```

2. Navigate into the project directory:
    ```cd trading```

3. Install the required dependencies:

---
## Usage
1. **Set Up API Keys:**
    - Obtain an API key for the **Oanda API** and configure it in the project.
    - Configure the **Gemini API** for financial news scraping.

2. **Run the Trading Agent**

3. **Telegram Bot Interaction:**

    You can interact with the agent through a Telegram bot. The bot provides real-time status updates about the agent's actions, such as buy, sell, or hold decisions, and displays the current state of training and testing.
---
## Features:
### **1. Dynamic Instrument Weight:**
The model dynamically adjusts the weight of the instruments based on their performance.
If an instrument makes the correct prediction, it is rewarded while others are penalized.
The instrument's weights start off equally distributed but are adjusted to focus on the most relevant instruments with weights normalized to 1.

### **2. Dynamic Decay:**
The rewards and punishment are governed by a decay function that follows a sigmoid curve. 
This allows for a more forgiving behavior in the earlier stages of training, but gradually reduces tolerance for errors as the model matures.

### **3. Q-Learning and Q-table:**
The agent is trained using Q-Learning, a reinforcement learning algorithm. A Q-table is used to store state-action values (hold, long and short). Enabling the agent to make decisions based on experience while improving
its performance over time.

### **4. Bayesian Optimization:**
The hyperparameters such as thresholds for technical indicators and reinforcement learning parameters (Alpha, Gamma, etc.) are optimized using the
Bayesian Optimization.

### **5. Selenium Web-scraping & Gemini API integration:**
Another feature that will be developed deeper helps to retrieve real-time news (randomly based on pre-defined prompts) news and financial data using
the Selenium web scraping. This will later affect how the agent makes trading decisions based on a Large Language Model (LLM).

### **6. Pair Plot and Line Graph Visualization:**
The model will display pair plots to visualize parameters relationship and line graphs to visualize reward performance over time.

### **7. Technical Indicators:**
The agent will make use of key technical indicators like:

    1. Simple Moving Average (SMA) 
    2. Exponential Moving Average (EMA) 12 & 26
    3. Relative Strength Index (RSI)
    4. Moving Average Convergence Divergence (MACD)
---

## Example output: 
- Call iteration: 1/1
- Alpha: 0.796
- Gamma: 0.755
- Decay: 3.449
- MACD Threshold: 0.0007
- Max Gradient: 0.094
- Scaling Factor: 0.058
- Gradient: 0.235
- Mid-Point: 23.34
- [Parameters pair-plot graph]
- [Reward line-plot graph]

---
## Contributing

Contributions are welcome! If you want to contribute, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.