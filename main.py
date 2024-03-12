import multiprocessing
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# Binance API
from binance.client import Client
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from itertools import product

multiprocessing.set_start_method('fork')

load_dotenv()

# Binance API and Secret Keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
enable_cache = False
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
btc_price_df_cache_file = str(start_date.year) + '-btc_price_df_cache.pkl'
if os.path.exists(btc_price_df_cache_file) and enable_cache:
    btc_price_df = joblib.load(btc_price_df_cache_file)
else:
    btc_price_df = get_btc_price_df(start_date, datetime.now(), Client.KLINE_INTERVAL_1DAY)
    joblib.dump(btc_price_df, btc_price_df_cache_file)


#################################################################################################################
    
def apply_strategy_just_hold(nothing, initial_owned_usdt):
    # print(f"Applying strategy with initial_owned_usdt={initial_owned_usdt}")
    # just hold the initial amount of BTC that can be bought with the initial USDT
    initial_btc_price = btc_price_df['close'].iloc[0]
    initial_btc_quantity = initial_owned_usdt / initial_btc_price
    final_portfolio_value = initial_btc_quantity * btc_price_df['close'].iloc[-1]
    portfolio_history = []
    buy_signals = []
    buy_signals.append(btc_price_df.index[0])

    for timestamp, current_price in btc_price_df['close'].items():
        portfolio_history.append((timestamp, initial_btc_quantity * current_price))
    portfolio_history_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'value']).set_index('timestamp')
    final_portfolio_value = portfolio_history_df['value'].iloc[-1]
    return final_portfolio_value, portfolio_history_df, buy_signals, []

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

def apply_strategy_with_volume_and_macd(sell_percentage, buy_percentage, initial_owned_usdt, volume_factor, macd_fast, macd_slow, macd_signal):
    previous_price = btc_price_df['close'].iloc[0]
    previous_volume = btc_price_df['volume'].iloc[0]
    last_action = "HOLD"
    quantity_owned_btc = 0
    quantity_owned_usdt = initial_owned_usdt
    portfolio_history = []
    buy_signals = []
    sell_signals = []

    # Calculate MACD components
    exp1 = btc_price_df['close'].ewm(span=macd_fast, adjust=False).mean()
    exp2 = btc_price_df['close'].ewm(span=macd_slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    for timestamp, row in btc_price_df.iterrows():
        current_price = row['close']
        current_volume = row['volume']
        current_macd = macd.loc[timestamp]
        current_signal = signal.loc[timestamp]
        price_change = (current_price - previous_price) / previous_price
        volume_change = (current_volume - previous_volume) / previous_volume if previous_volume else 0

        # Check conditions for buy and sell signals based on price, volume change, and MACD crossing the signal line
        if last_action != "SELL" and quantity_owned_btc > 0 and ((current_macd < current_signal and price_change < 0 and abs(price_change) >= sell_percentage and volume_change > volume_factor) or (current_macd < current_signal and current_macd < 0)):
            last_action = "SELL"
            quantity_owned_usdt = quantity_owned_btc * current_price
            quantity_owned_btc = 0
            sell_signals.append(timestamp)

        elif last_action != "BUY" and quantity_owned_usdt > 0 and ((current_macd > current_signal and price_change > 0 and price_change >= buy_percentage and volume_change > volume_factor) or (current_macd > current_signal and current_macd > 0)):
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

def execute_strategy(strategy_func, params):
    return strategy_func(*params)

# def optimize_strategy(strategy_func, parameter_space):
#     optimization_results = []
#     for params in product(*parameter_space):
#         final_portfolio_value, _, _, _ = strategy_func(*params)
#         optimization_results.append((params, final_portfolio_value))
#     best_parameters = max(optimization_results, key=lambda x: x[1])
#     return best_parameters

def strategy_wrapper(args):
    """
    Wrapper function to call the strategy function with a set of parameters.
    This is necessary for multiprocessing to unpack the arguments correctly.
    """
    strategy_func, params = args
    return strategy_func(*params), params

def optimize_strategy(strategy_func, parameter_space):
    optimization_results = []
    
    with ProcessPoolExecutor() as executor:
        # Prepare a list of arguments for the strategy_wrapper function
        parameters = product(*parameter_space)
        parameters_list = list(product(*parameter_space))
        print("Submitting tasks for optimization. Total tasks:", len(parameters_list))
        # print number of arguments to be passed to the strategy_wrapper function
        print("Number of arguments passed to the strategy:", len(parameter_space))
        futures = [executor.submit(strategy_wrapper, (strategy_func, params))
                   for params in parameters]
        print("Submitted tasks for optimization.")
        for future in as_completed(futures):
            try:
                if future.done():
                    result, params = future.result()
                    if result is not None:
                        final_portfolio_value = result[0]  # Unpack the first element of the result tuple
                        optimization_results.append((params, final_portfolio_value))
                        # print(f"Task completed successfully for parameters: {params}")
                    else:
                        print(f"Warning: Task completed but returned None for parameters: {params}")
                else:
                    print("Error: Future did not complete successfully.")
            except Exception as e:
                # Print the exception for debugging purposes
                print(f"An error occurred: {e}")

    if optimization_results:
        # Find the best parameter set
        best_parameters = max(optimization_results, key=lambda x: x[1])
        worst_parameters = min(optimization_results, key=lambda x: x[1])
        return (best_parameters, worst_parameters)
    else:
        print("No optimization results were produced.")
        return None

def main():
    # Define parameter ranges for testing
    sell_percentage_range = np.arange(0.01, 0.20, 0.01)  # Testing from 1% to 5% in 1% increments
    buy_percentage_range = np.arange(0.01, 0.20, 0.01)   # Testing from 1% to 5% in 1% increments
    initial_owned_usdt_range = np.array([100]) #np.arange(100, 1000, 100)  # Testing from 100 to 1000 in 100 increments
    volume_factor_range = np.arange(0.01, 0.2, 0.05)
    macd_fast_range = np.arange(8, 17, 3)
    macd_slow_range = np.arange(20, 31, 3)
    macd_signal_range = np.arange(6, 13, 2)

    strategies = {
        # "strategy_0": (apply_strategy_just_hold, "Only hold", (np.array([0]), initial_owned_usdt_range)),
        # "strategy_1": (apply_strategy, "Simple up/down", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range)),
        # "strategy_2": (apply_strategy_with_volume, "Up/Down with volume", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range, volume_factor_range)),
        # "strategy_3": (apply_strategy_with_volume_and_macd, "Up/Down+Volume+MACD", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range, volume_factor_range, macd_fast_range, macd_slow_range, macd_signal_range))        
    }
    
    strategy_best_results = {}
    strategy_worst_results = {}

    for name, (strategy_func, description, param_ranges) in strategies.items():
        # Unpack the parameter ranges as needed for each strategy
        best_parameters = optimize_strategy(strategy_func, param_ranges)
        strategy_best_results[name] = (description, best_parameters[0][0], execute_strategy(strategy_func, best_parameters[0][0]))
        strategy_worst_results[name] = (description, best_parameters[1][0], execute_strategy(strategy_func, best_parameters[1][0]))

    # manually set params for a strategy
    my_params = (0.09, 0.11, 100, 0.11, 11, 23, 12)
    strategy_best_results['strategy_99'] = ('Up/Down+Volume+MACD', my_params, execute_strategy(apply_strategy_with_volume_and_macd, my_params))


    if os.path.exists(str(start_date.year) + '-strategy_best_results.pkl'):
        previous_strategy_best_results = joblib.load(str(start_date.year) + '-strategy_best_results.pkl')
        strategy_best_results = {**previous_strategy_best_results, **strategy_best_results}
    if os.path.exists(str(start_date.year) + '-strategy_worst_results.pkl'):
        previous_strategy_worst_results = joblib.load(str(start_date.year) + '-strategy_worst_results.pkl')
        strategy_worst_results = {**previous_strategy_worst_results, **strategy_worst_results}
    # save the best results
    joblib.dump(strategy_best_results, str(start_date.year) + '-strategy_best_results.pkl')
    joblib.dump(strategy_worst_results, str(start_date.year) + '-strategy_worst_results.pkl')

    # Print the best results
    for name, (description, params, (final_portfolio_value, portfolio_history_df, _, _)) in strategy_best_results.items():
        print(f"Best parameters for {name}: {params} with final portfolio value: {final_portfolio_value:.2f} USDT")


if __name__ == "__main__":
    main()