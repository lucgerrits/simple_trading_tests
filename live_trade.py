# This program is used to simulate live trading using current data.

import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

# Binance API and Secret Keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

def main():
    # Initialize the Binance client
    client = Client(api_key, api_secret)

    # Step 1: get last 3 days of 6h candles

    # Step 2: listen to the websocket for new candles

    # Step 3: for each new candle, predict the price change and buy/sell accordingly

    # Step 4: update the portfolio value and save the results

    # Step 5: plot the results live 