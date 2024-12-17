PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

CALLS: int = 1
EPISODES: int = 5
RANDOM_STATE: int = 42
OMIT_ROWS: int = 1  # Minimum 1

EMA_PERIODS: list[int] = [12, 26]
LIVE_INSTR: list[str] = ["USD_JPY,EUR_USD"]
STATE_MAP: dict = {"Long": 0, "Short": 1, "Ignore": 2}

AVAIL_ACTIONS: list[str] = ["Long", "Short", "Ignore"]
AVAIL_INSTR: list[str] = ["MACD", "EMA"]
AVAIL_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]
AVAIL_GRAPHS: list[str] = ["Pair Plot", "Line Plot"]

OPTIMIZE_PARAM: list[float] = [0.7965429868602331,
                               0.7550304369598491,
                               3.449227500681924,
                               0.000798425078973,
                               0.09470822304218234,
                               0.058987708750821426,
                               0.23503195706327495,
                               23.348344445560876]

PARAM_NAME: list[str] = ["Alpha", "Gamma", "Decay", "MACD Threshold",
                         "Max Gradient", "Scaling Factor", "Gradient", "Mid-Point"]

PARAM_TRAINING: list[tuple] = [(0.00, 1.00), (0.70, 1.00), (1.50, 4.00), (0.0005, 0.001),
                               (0.01, 0.20), (0.01, 0.50), (0.01, 0.50), (10.0, 50.0)]




