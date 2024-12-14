
import json
import requests
import oandapyV20
client = oandapyV20.API(access_token="6da0bca9c2e9cdc2199d29c3a2e29bb9-779e90b35ada83790293d8bb47247683")

ACCESS_TOKEN = "6da0bca9c2e9cdc2199d29c3a2e29bb9-779e90b35ada83790293d8bb47247683"  # Replace with your OANDA API key
ACCOUNT_ID = "101-003-24666819-001"  # Replace with your OANDA account ID
STREAM_URL = "https://stream-fxpractice.oanda.com/v3/accounts"  # For practice
# Use "https://stream-fxtrade.oanda.com/v3/accounts" for live trading

def stream_prices(instruments):
    """
    Stream live FX prices from OANDA.
    :param instruments: A comma-separated string of instrument symbols (e.g., 'EUR_USD,USD_JPY').
    """
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    params = {
        "instruments": instruments  # Specify the instruments to stream
    }

    try:
        with requests.get(f"{STREAM_URL}/{ACCOUNT_ID}/pricing/stream", headers=headers, params=params,
                          stream=True) as response:
            if response.status_code != 200:
                print(f"Error: Unable to connect to stream. Status Code: {response.status_code}")
                print(response.json())
                return

            print("Streaming live FX prices...")
            for line in response.iter_lines():
                if line:
                    # Decode the JSON data from the stream
                    data = json.loads(line.decode("utf-8"))
                    print(json.dumps(data, indent=4))  # Pretty print the data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    instruments_to_stream = "USD_JPY"
    stream_prices(instruments_to_stream)
