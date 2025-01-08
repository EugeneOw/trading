import logging
import random
from constants import constants as c

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class StateManager:
    def __init__(self, epsilon, macd_margin, ema_margin, sma_margin, rsi_margin, smab_margin):
        self.epsilon: float = epsilon

        self.macd_margin: float = macd_margin
        self.ema_margin: float = ema_margin
        self.sma_margin: float = sma_margin
        self.rsi_margin: float = rsi_margin
        self.smab_margin: float = smab_margin

        self.macd, self.signal_line = 0, 0
        self.ema_26, self.ema_12 = 0, 0
        self.mid_price, self.sma = 0, 0
        self.low_band, self.upp_band = 0, 0
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
            self.upp_band = curr_row['Upper Band']

            self.rsi = curr_row['RSI']

            # Selects state randomly
            if random.uniform(0, 1) < decay:
                random.choice([self.macd_state, self.ema_state, self.sma_state,
                               self.rsi_state, self.smab_state])()
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
                elif highest_score == 3:
                    self.rsi_state()
                else:
                    self.smab_state()

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
        MACD interpretation:
            Long signal: MACD > Signal line
            Short signal: MACD < Signal line
        """
        self.instrument = self.avail_instr.index("MACD")

        macd_margin_difference = (self.macd - self.signal_line)
        macd_margin_threshold = (self.signal_line * (self.macd_margin / 100))

        if (self.macd > self.signal_line) and (macd_margin_difference <= macd_margin_threshold):
            print("MACD - Bearish signal")
            self.action = self.state_map[self.avail_actions[0]]
        elif self.macd < self.signal_line and (macd_margin_difference >= macd_margin_threshold):
            print("MACD - Bullish signal")
            self.action = self.state_map[self.avail_actions[1]]
        else:
            print("MACD - Ignore signal")
            self.action = self.state_map[self.avail_actions[2]]

    def ema_state(self):
        """
        Performs calculations and determines if the current row belongs to Buy or Sell state.
        EMA interpretation:
            Long signal: EMA-12 > EMA 26
            Short signal: EMA-12 < EMA 26
        """
        self.instrument = self.avail_instr.index("EMA")

        ema_difference = (self.ema_12 - self.ema_26)
        ema_margin_difference = ema_difference - self.mid_price
        ema_margin_threshold = (ema_difference * (self.ema_margin / 100))

        if (ema_difference > self.mid_price) and (ema_margin_difference <= ema_margin_threshold):
            print("EMA - Bearish signal")
            self.action = self.state_map[self.avail_actions[0]]
        elif (ema_difference < self.mid_price) and (ema_margin_difference >= ema_margin_threshold):
            print("EMA - Bullish signal")
            self.action = self.state_map[self.avail_actions[1]]
        else:
            print("EMA - Ignore signal")
            self.action = self.state_map[self.avail_actions[2]]

    def sma_state(self,):
        """
        Performs calculation and determines if the current row belongs to Buy or Sell state.
        SMA interpretation:
            Buy signal: Price > SMA
            Sell signal: Price < SMA
        """
        self.instrument = self.avail_instr.index("SMA")

        sma_margin_difference = self.mid_price-self.sma
        sma_margin_threshold = (self.sma*(self.sma_margin/100))

        if self.mid_price > self.sma and sma_margin_difference <= sma_margin_threshold:
            print("SMA - Bearish signal")
            self.action = self.state_map[self.avail_actions[0]]
        elif self.mid_price < self.sma and sma_margin_difference >= sma_margin_threshold:
            print("SMA - Bullish signal")
            self.action = self.state_map[self.avail_actions[1]]
        else:
            print("SMA - Ignore signal")
            self.action = self.state_map[self.avail_actions[2]]

    def rsi_state(self):
        """
        Performs calculation and determines if the current row belongs to Buy or Sell state.
        RSI interpretation:
            Buy signal: RSI < 30
            Sell signal: RSI > 70
        """
        self.instrument = self.avail_instr.index("RSI")

        rsi_margin_threshold = self.rsi*(self.rsi_margin/100)

        if (self.rsi < 30) and ((30-self.rsi) <= rsi_margin_threshold):
            print("RSI - Bearish signal")
            self.action = self.state_map[self.avail_actions[0]]
        elif self.rsi > 30 and ((self.rsi-70) <= rsi_margin_threshold):
            print("RSI - Bullish signal")
            self.action = self.state_map[self.avail_actions[1]]
        else:
            print("RSI - Ignore signal")
            self.action = self.state_map[self.avail_actions[2]]

    def smab_state(self):
        """
        Performs calculation and determines if the current row belongs to Buy or Sell state.
        SMAB interpretation:
            Buy signal: SMA > upper band
            Long signal: SMA < lower band
        """
        self.instrument = self.avail_instr.index("SMAB")
        if self.sma > self.upp_band:
            print("SMAB - Bearish signal")
            self.action = self.state_map[self.avail_actions[0]]
        elif self.sma < self.low_band:
            print("SMAB - Bullish signal")
            self.action = self.state_map[self.avail_actions[1]]
        else:
            print("SMAB - Ignore signal")
            self.action = self.state_map[self.avail_actions[2]]