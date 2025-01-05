# Trading Agent with Reinforcement Learning and Optimization
## Overview
This repository contains the code for an agent that can:

 1.	```\build``` (Re)build a dataset with real-time financial news and indicators.
 2.	```\optimize``` Optimize training parameters (Bayesian optimization).
 3.	```\train``` Train using reinforcement learning (Q-learning).
 4.	```\test``` Test on real-time data (Oanda API).
 5.	```\news``` Gather real-time financial news (Selenium web-scrapping + Gemini API).

The agent will provide real-time status updates and results through a Telegram Bot.

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
    ```pip install -r requirements.txt```

---
## Usage
1. **Set Up API Keys:**
    - Obtain an API key for the **Oanda API** and configure it in the project.
    - Configure the **Gemini API** for financial news scraping.

2. **Run the Trading Agent:**

   To start the agent, run the following script:
   ```python agent.py```

3. **Telegram Bot Interaction:**

    You can interact with the agent through a Telegram bot. The bot provides real-time status updates about the agent's actions, such as buy, sell, or hold decisions, and displays the current state of training and testing.
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