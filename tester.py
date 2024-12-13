import requests
import json
import time
from datetime import datetime, timezone
import pytz  # Install with `pip install pytz`

# TraderMade API endpoint
BASE_URL = "https://marketdata.tradermade.com/api/v1"
API_KEY = "pxosn6IpIAmhMEOyTCKf"


def get_live_fx_data(symbol: str):
    """
    Fetch live FX data for a given currency pair.
    """
    endpoint = f"{BASE_URL}/live"
    params = {"currency": symbol, "api_key": API_KEY}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = json.loads(response.content.decode("utf-8"))

        # Explicitly check for required keys
        if "quotes" in data and "timestamp" in data:
            quote = data["quotes"][0]
            ask = quote["ask"]
            bid = quote["bid"]
            mid = quote["mid"]
            timestamp = data["timestamp"]
            return ask, bid, mid, timestamp
        else:
            print("Invalid data structure or missing keys in response.")
            return None, None, None, None

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None, None, None, None


def convert_to_sgt(timestamp):
    """
    Convert a UTC timestamp to Singapore Time (SGT).
    """
    try:
        # Convert Unix timestamp to datetime
        utc_time = datetime.fromtimestamp(timestamp, timezone.utc)

        # Convert to SGT timezone
        sgt_timezone = pytz.timezone("Asia/Singapore")
        sgt_time = utc_time.astimezone(sgt_timezone)
        return sgt_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error converting time: {e}")
        return None


if __name__ == "__main__":
    currency_pair = "USDJPY"
    update_interval = 5  # Fetch data every 5 seconds
    print(f"Fetching live data for {currency_pair} every {update_interval} seconds...")

    try:
        while True:
            ask, bid, mid, timestamp = get_live_fx_data(currency_pair)
            if ask is not None and bid is not None and timestamp is not None:
                sgt_time = convert_to_sgt(timestamp)
                print(f"SGT Time: {sgt_time}, \nBid: {bid}, Ask: {ask}, Mid: {mid}")
            else:
                print("Failed to fetch data or missing fields in the response.")
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("Stopped fetching live data.")