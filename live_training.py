import oandapyV20
import oandapyV20.endpoints as oandapi
import json


class LiveTraining:
    def __init__(self):
        self.account_id = ''
        self.access_token = ''
        self.client = oandapyV20.API(
            access_token=self.access_token)
        self.params = {
            "granularity": "1M",
            "count": 10,
        }

    def get_rates(self):
        try:
            response = self.client.request(oandapi.PricingInfo())
