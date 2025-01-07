import logging
import random
from constants import constants as c

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class StateManager:
    def __init__(self, macd_threshold, epsilon):
        self.epsilon: float = epsilon
        self.macd_threshold: float = macd_threshold

        self.macd, self.signal_line = 0, 0
        self.ema_26, self.ema_12 = 0, 0
        self.mid_price, self.sma, self.low_band = 0, 0, 0
        self.rsi = 0

        self.state_map = c.STATE_MAP
        self.avail_instr = c.AVAIL_INSTR
        self.avail_actions = c.AVAIL_ACTIONS

        self.action: str = ""
        self.instrument: int = 0

    @staticmethod
    def create_weights():
        """
        Creates a list that stores equally distributed number of weights.

        :return:
        :rtype:List[float]
        """
        number_of_instruments = len(c.AVAIL_INSTR)
        return [1/number_of_instruments]*number_of_instruments

    def define_state(self, curr_row, instr_weight, decay):
        """
        Defines the state of each row.
        The list that contains the weights of each instrument is used to determine
        which instrument is best suited for this but is not updated.
        It is updated only later when we calculate the
        reward.

        :param curr_row: Contains details of each row

        :param instr_weight: Contains weights of each instrument
        :type instr_weight: list[float]

        :param decay: Contains the value to be larger than to select a pre-defined action
        :type decay: float

        :return: Returns a string containing action
        :rtype: str
        """
        try:
            self.macd = curr_row["MACD"]
            self.signal_line = curr_row['Signal Line']

            self.ema_12 = curr_row[f'{"EMA"} {c.EMA_PERIODS[0]}']
            self.ema_26 = curr_row[f'{"EMA"} {c.EMA_PERIODS[1]}']

            self.sma = curr_row['SMA']
            self.mid_price = curr_row['Mid Price']
            self.low_band = curr_row['Lower Band']

            self.rsi = curr_row['RSI']

            # Selects state randomly
            if random.uniform(0, 1) < decay:
                random.choice([self.macd_state(), self.ema_state(), self.sma_state(), self.rsi_state()])
            else:
                if random.uniform(0, 1) < 0.1:
                    # 10% chance of selecting the least weighted instrument.
                    highest_score = instr_weight.index(min(instr_weight))
                else:
                    highest_score = instr_weight.index(max(instr_weight))

                if highest_score == 0:
                    self.macd_state()
                elif highest_score == 1:
                    self.ema_state()
                elif highest_score == 2:
                    self.sma_state()
                else:
                    self.rsi_state()
            return self.action, self.instrument

        except KeyError:
            logging.error("Row doesn't exists")
        except IndexError:
            logging.error("Attempting to access index that is out of bonds.")
        except Exception as e:
            logging.error("Unexpected error: ", e)
            raise

    def macd_state(self):
        """
        Performs calculations and determines if the current row belongs to Buy or Sell state.
        """
        self.instrument = self.avail_instr.index("MACD")
        if (self.macd > self.signal_line) and (self.macd - self.signal_line) < self.macd_threshold:
            self.action = self.state_map[self.avail_actions[0]]
        elif (self.macd < self.signal_line) and (self.signal_line - self.macd) < self.macd_threshold:
            self.action = self.state_map[self.avail_actions[1]]
        else:
            self.action = self.state_map[self.avail_actions[2]]

    def ema_state(self):
        """
        Performs calculations and determines if the current row belongs to Buy or Sell state.
        """
        self.instrument = self.avail_instr.index("EMA")
        if (self.ema_12 - self.ema_26) > 0:
            self.action = self.state_map[self.avail_actions[0]]
        elif (self.ema_12 - self.ema_26) < 0:
            self.action = self.state_map[self.avail_actions[1]]
        else:
            self.action = self.state_map[self.avail_actions[2]]

    def sma_state(self,):
        """
        Performs calculation and determines if the current row belongs to Buy or Sell state.
        """
        self.instrument = self.avail_instr.index("SMA")
        if self.mid_price > self.sma and self.mid_price > self.low_band:
            self.action = self.state_map[self.avail_actions[0]]
        elif self.mid_price < self.sma and self.mid_price < self.low_band:
            self.action = self.state_map[self.avail_actions[1]]
        else:
            self.action = self.state_map[self.avail_actions[2]]

    def rsi_state(self):
        """
        Performs calculation and determines if the current row belongs to Buy or Sell state.
        """
        self.instrument = self.avail_instr.index("RSI")
        if self.rsi < 30:
            self.action = self.state_map[self.avail_actions[0]]
        elif self.rsi > 30:
            self.action = self.state_map[self.avail_actions[1]]
        else:
            self.action = self.state_map[self.avail_actions[2]]
