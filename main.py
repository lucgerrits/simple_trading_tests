import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# Binance API
from binance.client import Client
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

load_dotenv()

# Binance API and Secret Keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
enable_cache = True
start_date = datetime(2020, 1, 1)
# Binance Client
client = Client(api_key, api_secret)

def get_btc_price_df(start, end, interval):
    btc_price = client.get_historical_klines("BTCUSDT", interval, start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
    btc_price_df = pd.DataFrame(btc_price, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    btc_price_df['timestamp'] = pd.to_datetime(btc_price_df['timestamp'], unit='ms')
    btc_price_df.set_index('timestamp', inplace=True)
    btc_price_df['close'] = btc_price_df['close'].astype(float)
    btc_price_df['open'] = btc_price_df['open'].astype(float)
    btc_price_df['high'] = btc_price_df['high'].astype(float)
    btc_price_df['low'] = btc_price_df['low'].astype(float)
    btc_price_df['volume'] = btc_price_df['volume'].astype(float)
    return btc_price_df

# Cache the BTC price data
btc_price_df_cache_file = 'btc_price_df_cache.pkl'
if os.path.exists(btc_price_df_cache_file) and enable_cache:
    btc_price_df = joblib.load(btc_price_df_cache_file)
else:
    btc_price_df = get_btc_price_df(start_date, datetime.now(), Client.KLINE_INTERVAL_1DAY)
    joblib.dump(btc_price_df, btc_price_df_cache_file)


#################################################################################################################
    
def apply_strategy(sell_percentage, buy_percentage, initial_owned_usdt):
    # print(f"Applying strategy with sell_percentage={sell_percentage}, buy_percentage={buy_percentage}, initial_owned_usdt={initial_owned_usdt}")
    previous_price = btc_price_df['close'].iloc[0]
    # Initialize variables to track the buy and sell signals
    last_action = "HOLD" # Start with HOLD, can be BUY or SELL
    quantity_owned_btc = 0 # Assuming starting with 1 BTC for the initial price of previous_price
    quantity_owned_usdt = initial_owned_usdt  # Assuming starting with 0 USDT for simplicity
    portfolio_history = []
    buy_signals = []
    sell_signals = []

    for timestamp, current_price in btc_price_df['close'].items():
        price_change = (current_price - previous_price) / previous_price

        if price_change < 0 and abs(price_change) >= sell_percentage and last_action != "SELL" and quantity_owned_btc > 0:
            last_action = "SELL"
            quantity_owned_usdt = quantity_owned_btc * current_price
            quantity_owned_btc = 0
            sell_signals.append(timestamp)

        elif price_change > 0 and price_change >= buy_percentage and last_action != "BUY" and quantity_owned_usdt > 0:
            last_action = "BUY"
            quantity_owned_btc = quantity_owned_usdt / current_price
            quantity_owned_usdt = 0
            buy_signals.append(timestamp)
        
        previous_price = current_price
        portfolio_history.append((timestamp, quantity_owned_btc * current_price + quantity_owned_usdt))
    
    portfolio_history_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'value']).set_index('timestamp')
    final_portfolio_value = portfolio_history_df['value'].iloc[-1]
    return final_portfolio_value, portfolio_history_df, buy_signals, sell_signals

def apply_strategy_with_volume(sell_percentage, buy_percentage, initial_owned_usdt, volume_factor):
    # print(f"Applying strategy with sell_percentage={sell_percentage}, buy_percentage={buy_percentage}, initial_owned_usdt={initial_owned_usdt}, volume_factor={volume_factor}")
    previous_price = btc_price_df['close'].iloc[0]
    previous_volume = btc_price_df['volume'].iloc[0]
    last_action = "HOLD"
    quantity_owned_btc = 0
    quantity_owned_usdt = initial_owned_usdt
    portfolio_history = []
    buy_signals = []
    sell_signals = []

    for timestamp, row in btc_price_df.iterrows():
        current_price = row['close']
        current_volume = row['volume']
        price_change = (current_price - previous_price) / previous_price
        volume_change = (current_volume - previous_volume) / previous_volume if previous_volume else 0

        # Use volume change to confirm buy or sell signal
        if price_change < 0 and abs(price_change) >= sell_percentage and volume_change > volume_factor and last_action != "SELL" and quantity_owned_btc > 0:
            last_action = "SELL"
            quantity_owned_usdt = quantity_owned_btc * current_price
            quantity_owned_btc = 0
            sell_signals.append(timestamp)

        elif price_change > 0 and price_change >= buy_percentage and volume_change > volume_factor and last_action != "BUY" and quantity_owned_usdt > 0:
            last_action = "BUY"
            quantity_owned_btc = quantity_owned_usdt / current_price
            quantity_owned_usdt = 0
            buy_signals.append(timestamp)
        
        previous_price = current_price
        previous_volume = current_volume
        portfolio_history.append((timestamp, quantity_owned_btc * current_price + quantity_owned_usdt))
    
    portfolio_history_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'value']).set_index('timestamp')
    final_portfolio_value = portfolio_history_df['value'].iloc[-1]
    return final_portfolio_value, portfolio_history_df, buy_signals, sell_signals

#################################################################################################################

# Function to execute for each combination of parameters
# def execute_strategy(params):
#     sell_percentage, buy_percentage, initial_owned_usdt = params
#     portfolio_history_df, buy_signals, sell_signals = apply_strategy(btc_price_df, sell_percentage, buy_percentage, initial_owned_usdt)
#     # portfolio_history_df, buy_signals, sell_signals = apply_strategy_with_volume(btc_price_df, sell_percentage, buy_percentage, initial_owned_usdt, 0.1)
#     final_portfolio_value = portfolio_history_df['value'].iloc[-1]
#     return (sell_percentage, buy_percentage, initial_owned_usdt, final_portfolio_value)

# def run_optimization(sell_percentage_range, buy_percentage_range, initial_owned_usdt_range):
#     # Flatten the parameter space to a list of tuples. Each tuple will represent a single combination of parameters.
#     parameters = [(sell_percentage, buy_percentage, initial_owned_usdt) 
#                   for sell_percentage in sell_percentage_range 
#                   for buy_percentage in buy_percentage_range 
#                   for initial_owned_usdt in initial_owned_usdt_range]

#     # Placeholder for storing the results
#     optimization_results = []

#     # Use ProcessPoolExecutor to run the strategies in parallel
#     with ProcessPoolExecutor() as executor:
#         futures = [executor.submit(execute_strategy, params) for params in parameters]
#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 optimization_results.append(result)
#             except Exception as exc:
#                 print(f'Generated an exception: {exc}')
    
#     return optimization_results

# # Run the optimization
# optimization_results = run_optimization(sell_percentage_range, buy_percentage_range, initial_owned_usdt_range)

# # Find the combination of parameters that resulted in the highest final portfolio value
# best_parameters = max(optimization_results, key=lambda x: x[3])

# print(f"Best parameters: Sell at {best_parameters[0]*100:.2f}%, Buy at {best_parameters[1]*100:.2f}%, Initial USDT: {best_parameters[2]:.2f}")
# print(f"Final portfolio value: {best_parameters[3]:.2f} USDT")

# # Apply the best strategy
# sell_percentage = best_parameters[0]
# buy_percentage = best_parameters[1]
# initial_owned_usdt = best_parameters[2]
# portfolio_history_df, buy_signals, sell_signals = apply_strategy(btc_price_df, sell_percentage, buy_percentage, initial_owned_usdt)



def execute_strategy(strategy_func, params):
    return strategy_func(*params)

def optimize_strategy(strategy_func, parameter_space):
    optimization_results = []
    for params in product(*parameter_space):
        final_portfolio_value, _, _, _ = strategy_func(*params)
        optimization_results.append((params, final_portfolio_value))
    best_parameters = max(optimization_results, key=lambda x: x[1])
    return best_parameters

# Define parameter ranges for testing
sell_percentage_range = np.arange(0.01, 0.50, 0.01)  # Testing from 1% to 5% in 1% increments
buy_percentage_range = np.arange(0.01, 0.50, 0.01)   # Testing from 1% to 5% in 1% increments
initial_owned_usdt_range = np.array([100]) #np.arange(100, 1000, 100)  # Testing from 100 to 1000 in 100 increments
volume_factor_range = np.array([0.1])

def main():
    strategies = {
        "strategy_1": (apply_strategy, (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range)),
        "strategy_2": (apply_strategy_with_volume, (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range, volume_factor_range)),
    }
    
    strategy_best_results = {}

    for name, (strategy_func, param_ranges) in strategies.items():
        # Unpack the parameter ranges as needed for each strategy
        best_parameters = optimize_strategy(strategy_func, param_ranges)
        strategy_best_results[name] = execute_strategy(strategy_func, best_parameters[0])
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    for name, (portfolio_history_df, _, _) in strategy_best_results.items():
        plt.plot(portfolio_history_df["timestamp"], portfolio_history_df['value'], label=name)
    
    plt.title('Comparison of Portfolio Performances for Different Strategies')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()











# # Plot all the data
# fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# # Plot BTC/USDT on the first subplot
# axs[0].plot(btc_price_df['close'], label='BTC/USDT')
# axs[0].set_title('BTC/USDT Price')

# # Plot the price change on the second subplot
# axs[1].plot(btc_price_df['close'].pct_change(), label='BTC/USDT % Change', color='orange')
# axs[1].title.set_text('BTC/USDT % Change')

# # Plot the portfolio value on the third subplot
# axs[2].plot(portfolio_history_df, label='Portfolio Value', color='green')
# axs[2].title.set_text('Portfolio Value with Buy/Sell Signals (Sell at {:.2f}%, Buy at {:.2f}%, and {} USDT)'.format(sell_percentage*100, buy_percentage*100, initial_owned_usdt))
# # Add buy and sell signals to the portfolio value plot
# axs[2].scatter(buy_signals, portfolio_history_df.loc[buy_signals, 'value'], label='Buy Signal', marker='^', color='blue')
# axs[2].scatter(sell_signals, portfolio_history_df.loc[sell_signals, 'value'], label='Sell Signal', marker='v', color='red')


# plt.tight_layout()
# plt.show()