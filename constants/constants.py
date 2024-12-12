
PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

EMA_PERIODS: list[int] = [12, 26]
STATE_MAP: dict = {"Long": 0, "Short": 1}

AVAILABLE_ACTIONS: list[str] = ["Long", "Short"]
AVAILABLE_INSTRUMENTS: list[str] = ["MACD", "EMA"]
AVAILABLE_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]

OPTIMIZE_PARAMETERS: list[float] = [1.0,
                                    0.7,
                                    0.1,
                                    1.0,
                                    1.0,
                                    -1.0,
                                    0.01,
                                    0.01,
                                    1.0,
                                    83.36485324122248]

PARAMETERS_NAME: list[str] = ["alpha",
                              "gamma",
                              "epsilon",
                              "decay",
                              "macd threshold",
                              "ema difference",
                              "max_gradient",
                              "scaling_factor",
                              "gradient",
                              "midpoint"]

PARAMETERS_TRAINING: list[tuple] = [(0.00, 1.00),
                                    (0.70, 1.00),
                                    (-1.0, 0.10),
                                    (1.00, 10.0),
                                    (-1.0, 1.00),
                                    (-1.0, 0.00),
                                    (0.01, 0.20),
                                    (0.01, 0.50),
                                    (0.01, 0.50),
                                    (10.0, 50.0)]
