PATH_DB: str = "../trading/file_paths.db"
Q_TABLE_DB: str = "../trading/q_table.db"

WEBPAGE: str = "https://www.google.com"
TIME_OUT: int = 5  # seconds before time-out
ARTICLES: int = 5  # Number of articles to summarise

CALLS: int = 1  # Changes parameters
EPISODES: int = 5  # Extends the test / cycle
OMIT_ROWS: int = 810001  # Minimum 1
RANDOM_STATE: int = 42

RSI_PERIODS: list[int] = [14]
SMA_PERIODS: list[int] = [20]
EMA_PERIODS: list[int] = [12, 26]
LIVE_INSTR: list[str] = ["USD_JPY,EUR_USD"]
STATE_MAP: dict = {"Long": 0, "Short": 1, "Ignore": 2}

AVAIL_INSTR: list[str] = ["MACD", "EMA", "B-Bands", "RSI"]
AVAIL_GRAPHS: list[str] = ["Pair Plot", "Line Plot"]
AVAIL_ACTIONS: list[str] = ["Long", "Short", "Ignore"]
AVAIL_STATES: list[str] = ["Bullish", "Bearish", "Neutral"]


PARAM_NAME: list[str] = ["Alpha", "Gamma", "Decay", "MACD Threshold",
                         "Max Gradient", "Scaling Factor", "Gradient", "Mid-Point"]

PARAM_TRAINING: list[tuple] = [(0.00, 1.00), (0.70, 1.00), (1.50, 4.00), (0.0005, 0.001),
                               (0.01, 0.20), (0.01, 0.50), (0.01, 0.50), (10.0, 50.0)]

OPTIMIZE_PARAM: list[float] = [0.00711,
                               0.9686,
                               2.4743,
                               0.0009,
                               0.1348,
                               0.4403,
                               0.4889,
                               48.474]

PROMPT: str = ("Please visit the website link below and provide a 30-second summary in paragraph form. Focus on the key points of the content and "
               "offer a brief overview. Avoid including unnecessary details, but ensure the response is informative and concise.If you’re unable to "
               "access the link, please reply ‘Error'")

