
PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

EMA_PERIODS: list[int] = [12, 26]
STATE_MAP: dict = {"Long": 0, "Short": 1, "Ignore": 2}

AVAIL_ACTIONS: list[str] = ["Long", "Short", "Ignore"]
AVAIL_INSTR: list[str] = ["MACD", "EMA"]
AVAIL_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]
AVAIL_GRAPHS: list[str] = ["pair plot", "line plot"]
OPTIMIZE_PARAM: list[float] = [0.7965429868602331,
                                    1.0,
                                    0,
                                    6.371651421518384,
                                    -0.1083344942928175,
                                    -0.9000250841819971,
                                    0.09725728947351478,
                                    0.17351721945812074,
                                    0.08000474078175099,
                                    36.03553891795412]

PARAM_NAME: list[str] = ["alpha",
                              "gamma",
                              "decay",
                              "macd threshold",
                              "max_gradient",
                              "scaling_factor",
                              "gradient",
                              "midpoint"]

PARAM_TRAINING: list[tuple] = [(0.00, 1.00),  # alpha
                                    (0.70, 1.00),  # gamma
                                    (1.50, 4.00),  # decay / epsilon
                                    (0.0005, 0.001),  # macd threshold
                                    (0.01, 0.20),  # max gradient
                                    (0.01, 0.50),  # scaling
                                    (0.01, 0.50),  # gradient
                                    (10.0, 50.0)]  # mid point
