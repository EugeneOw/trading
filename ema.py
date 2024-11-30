import pandas as pd

#  ewm: Exponential weighted moving average (EWMA)
#  span: Controls how much past data has influence in recent data points.
#        Since span = 12, only past 12 data points has effect.


class EMA:
    def __init__(self, file):
        self.periods = [12, 24]
        self.df = pd.read_csv(file)

    def ema(self):
        self.df['Mid Price'] = (self.df['Ask'] + self.df['Bid']/2)
        for period in self.periods:
            self.df[f'EMA {period}'] = self.df['Mid Price'].ewm(span=period, adjust=False).mean()
        return self.df

    def macd(self):
        self.ema()
        self.df[f'MACD'] = self.df['EMA 24'] - self.df['EMA 12']
        return self.df
