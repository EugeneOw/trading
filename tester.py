from tradermade import stream

api_key = "pxosn6IpIAmhMEOyTCKf"

stream.set_ws_key(api_key)
stream.set_symbols("USDJPY")

def print_message(data):
    print(f"recevied {data}")

stream.stream_data(print_message)
stream.connect()

