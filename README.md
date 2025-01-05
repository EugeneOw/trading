# Trading Agent with Reinforcement Learning and Optimization
## Overview
This repository contains the code for an agent that can:

	1.	Combine, clean, and (re)build a dataset with real-time financial news and indicators,
	2.	Optimize training parameters (Bayesian optimization),
	3.	Train using reinforcement learnning (Q-learning),
	4.	Test on real-time data (Oanda API),
	5.	Gather and utilize real-time financial news (Selenium web-scrapping + Gemini API)

The agent will provide real-time status updates and results through a Telegram Bot.

---
### Requirements
  - Python 3.6 or higher
  - Required libraries:
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
---

### Example output: 
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
