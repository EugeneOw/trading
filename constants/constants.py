
EMA_PERIODS = [12, 26]
STATE_MAP = {"Buy": 0, "Sell": 1, "Hold": 2}
AVAILABLE_ACTIONS: list = ["Buy", "Sell", "Hold"]
PARAMETERS_NAME: list[str] = ["alpha", "gamma", "epsilon", "decay", "macd threshold", "ema difference"]

OPTIMIZE_PARAMETERS: list[float] = [1.0, 0.7,
                                    -0.5638253578639407,
                                    1.2923527222075457,
                                    0.8880416279011167,
                                    -0.2679725669702262,]

PATH_DB = "../trading/file_paths.db"
Q_TABLE_DB = "../trading/q_table.db"
