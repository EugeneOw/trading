
EMA_PERIODS: list[int] = [12, 26]

RANDOM_STATE: int = 42
OMITTED_ROWS: int = 810001  # Min 1
NO_OF_EPISODES: int = 10
NO_OF_CALLS: int = 50

PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

STATE_MAP: dict = {"Buy": 0, "Sell": 1, "Hold": 2}
AVAILABLE_ACTIONS: list = ["Buy", "Sell", "Hold"]
PARAMETERS_NAME: list[str] = ["alpha", "gamma", "epsilon", "decay", "macd threshold", "ema difference"]

OPTIMIZE_PARAMETERS: list[float] = [1.0, 0.7,
                                    -0.5638253578639407,
                                    1.2923527222075457,
                                    0.8880416279011167,
                                    -0.2679725669702262, ]


PARAMETERS_TRAINING: list[tuple] = [(0.00, 1.00),
                                    (0.70, 1.00),
                                    (-1.0, 0.10),
                                    (1.00, 10.0),
                                    (-1.0, 1.00),
                                    (-1.0, 0.00), ]
