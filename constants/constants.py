
PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

EMA_PERIODS: list[int] = [12, 26]
STATE_MAP: dict = {"Long": 0, "Short": 1}
constant_alpha: float = 0.00001  # Punishes or rewards the instrument for a correct/wrong selection

AVAILABLE_ACTIONS: list[str] = ["Long", "Short"]
AVAILABLE_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]
AVAILABLE_INSTRUMENTS: list[str] = ["MACD", "EMA"]

PARAMETERS_NAME: list[str] = ["alpha", "gamma", "epsilon", "decay", "macd threshold", "ema difference"]
OPTIMIZE_PARAMETERS: list[float] = [1.0, 0.99, 0.01, 0.1, 0.9675643535242828, 0]
PARAMETERS_TRAINING: list[tuple] = [(0.00, 1.00), (0.70, 1.00), (-1.0, 0.10),
                                    (1.00, 10.0), (-1.0, 1.00), (-1.0, 0.00), ]
