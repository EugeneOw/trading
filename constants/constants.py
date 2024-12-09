
EMA_PERIODS: list[int] = [12, 26]

PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

AVAILABLE_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]
STATE_MAP: dict = {"Buy": 0, "Sell": 1, "Hold": 2}
AVAILABLE_ACTIONS: list = ["Buy", "Sell", "Hold"]
PARAMETERS_NAME: list[str] = ["alpha", "gamma", "epsilon", "decay", "macd threshold", "ema difference"]

OPTIMIZE_PARAMETERS: list[float] = [1.0, 0.99, 0.01, 0.1, 0.9675643535242828, 0]


PARAMETERS_TRAINING: list[tuple] = [(0.00, 1.00),
                                    (0.70, 1.00),
                                    (-1.0, 0.10),
                                    (1.00, 10.0),
                                    (-1.0, 1.00),
                                    (-1.0, 0.00), ]
