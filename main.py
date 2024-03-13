import multiprocessing
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from binance.client import Client
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from strategies.strategies import Strategies
from strategies.SVMstrategies import SVMstrategies
from strategies.RFStrategies import RFstrategies

multiprocessing.set_start_method('fork')

load_dotenv()

# Binance API and Secret Keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

ml_enable_cache = True
ml_enable_train_cache = True
ml_start_date = datetime(2013, 1, 1) # the cache train the model from 2013
ml_cache_prefix = str(ml_start_date.year) + '-' + str(ml_start_date.month) + '-' + str(ml_start_date.day) + '-'
ml_stop_date = datetime(2022, 1, 1)

enable_cache = True
enable_basic_strat_opti_cache = True
start_date = datetime(2022, 1, 1)
cache_prefix = str(start_date.year) + '-' + str(start_date.month) + '-' + str(start_date.day) + '-'
stop_date = datetime.now()

klines_interval = Client.KLINE_INTERVAL_6HOUR
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

def calculate_idicator_values(btc_price_df):
    btc_price_df['macd'] = btc_price_df['close'].ewm(span=12, adjust=False).mean() - btc_price_df['close'].ewm(span=26, adjust=False).mean()
    btc_price_df['signal'] = btc_price_df['macd'].ewm(span=9, adjust=False).mean()
    btc_price_df['histogram'] = btc_price_df['macd'] - btc_price_df['signal']
    btc_price_df['rsi'] = 100 - (100 / (1 + (btc_price_df['close'] / btc_price_df['close'].shift(1))))
    btc_price_df['sma_20'] = btc_price_df['close'].rolling(window=20).mean()
    btc_price_df['sma_50'] = btc_price_df['close'].rolling(window=50).mean()
    btc_price_df['sma_200'] = btc_price_df['close'].rolling(window=200).mean()
    btc_price_df['ema_20'] = btc_price_df['close'].ewm(span=20, adjust=False).mean()
    btc_price_df['ema_50'] = btc_price_df['close'].ewm(span=50, adjust=False).mean()
    btc_price_df['ema_200'] = btc_price_df['close'].ewm(span=200, adjust=False).mean()
    btc_price_df['close_pct_change'] = btc_price_df['close'].pct_change()

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

def store_results(strategy_best_results, strategy_worst_results):
    # Load previous results and update them
    if os.path.exists(cache_prefix + 'strategy_best_results.pkl'):
        previous_strategy_best_results = joblib.load(cache_prefix + 'strategy_best_results.pkl')
        strategy_best_results = {**previous_strategy_best_results, **strategy_best_results}
    if os.path.exists(cache_prefix + 'strategy_worst_results.pkl'):
        previous_strategy_worst_results = joblib.load(cache_prefix + 'strategy_worst_results.pkl')
        strategy_worst_results = {**previous_strategy_worst_results, **strategy_worst_results}

    # Save the updated results
    joblib.dump(strategy_best_results, cache_prefix + 'strategy_best_results.pkl')
    joblib.dump(strategy_worst_results, cache_prefix + 'strategy_worst_results.pkl')
    return strategy_best_results, strategy_worst_results

def format_params(t):
    tb = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, t))
    # join all elements in the tuple with a comma and a space and bracket the result with parentheses
    return '(' + ', '.join(map(str, tb)) + ')'

def main():
    # Cache the BTC price data
    if os.path.exists(cache_prefix + 'btc_price_df_cache.pkl') and enable_cache:
        btc_price_df = joblib.load(cache_prefix + 'btc_price_df_cache.pkl')
    else:
        btc_price_df = get_btc_price_df(start_date, stop_date, klines_interval)
        # Calculate the indicator values
        calculate_idicator_values(btc_price_df)
        joblib.dump(btc_price_df, cache_prefix + 'btc_price_df_cache.pkl')
    # cache the ML model data
    if os.path.exists(ml_cache_prefix + 'btc_price_df_cache.pkl') and ml_enable_cache:
        ml_btc_price_df = joblib.load(ml_cache_prefix + 'btc_price_df_cache.pkl')
    else:
        ml_btc_price_df = get_btc_price_df(ml_start_date, ml_stop_date, klines_interval)
        # Calculate the indicator values
        calculate_idicator_values(ml_btc_price_df)
        joblib.dump(ml_btc_price_df, ml_cache_prefix + 'btc_price_df_cache.pkl')
        
    # print all columns names
    # print(btc_price_df.columns)

    # Create an instance of the Strategies class
    basic_strategies = Strategies(ml_btc_price_df) # optimize the strategies with the ml_btc_price_df

    strategy_best_results = {}
    strategy_worst_results = {}
    strategies_to_optimize = {}
    strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)

    # Define parameter ranges for testing
    initial_owned_usdt_range = np.array([100]) #np.arange(100, 1000, 100)  # Testing from 100 to 1000 in 100 increments
    sell_percentage_range = np.arange(0.001, 0.05, 0.001)  # Testing from 1% to 5% in 1% increments
    buy_percentage_range = np.arange(0.001, 0.05, 0.001)   # Testing from 1% to 5% in 1% increments
    volume_factor_range = np.arange(0.01, 0.1, 0.05)
    macd_fast_range = np.arange(8, 17, 3)
    macd_slow_range = np.arange(20, 31, 4)
    macd_signal_range = np.arange(6, 13, 3)
    strategies_to_optimize = {
        "strategy_0": (basic_strategies.apply_strategy_just_hold, "Only hold", (initial_owned_usdt_range,)),
        "strategy_1": (basic_strategies.apply_strategy, "Simple up/down", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range)),
        "strategy_2": (basic_strategies.apply_strategy_with_volume, "Up/Down with volume", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range, volume_factor_range)),
        # this one takes too long to run:
        # "strategy_3": (basic_strategies.apply_strategy_with_volume_and_macd, "Up/Down+Volume+MACD", (sell_percentage_range, buy_percentage_range, initial_owned_usdt_range, volume_factor_range, macd_fast_range, macd_slow_range, macd_signal_range))        
    }

    if strategies_to_optimize:
        for name, (strategy_func, description, param_ranges) in strategies_to_optimize.items():
            if name not in strategy_best_results or not enable_basic_strat_opti_cache:
                # Unpack the parameter ranges as needed for each strategy
                best_parameters = optimize_strategy(strategy_func, param_ranges)
                # run results with the btc_price_df
                # find a way to run the results with the btc_price_df and not the ml_btc_price_df
                strategy_func.__self__.btc_price_df = btc_price_df
                strategy_best_results[name] = (description, best_parameters[0][0], execute_strategy(strategy_func, best_parameters[0][0]))
                strategy_worst_results[name] = (description, best_parameters[1][0], execute_strategy(strategy_func, best_parameters[1][0]))
                strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)

    # manually set params for a strategy
    # my_params = (0.09, 0.11, 100, 0.11, 11, 23, 12)
    # strategy_best_results['strategy_99'] = ('Up/Down+Volume+MACD', my_params, execute_strategy(basic_strategies.apply_strategy_with_volume_and_macd, my_params))
    
    # Add ML strategies
    if 'strategy_svm_0' not in strategy_best_results:
        features = ['close', 'volume', 'macd', 'close_pct_change']
        ml_svn_strategies_0 = SVMstrategies(ml_btc_price_df, "0_" + ml_cache_prefix, features)
        ml_svn_strategies_0.maybe_train_model(ml_enable_train_cache)
        ml_svn_strategies_0.btc_price_df = btc_price_df # Use the same btc_price_df for the ML strategies as the basic strategies
        my_params = (100,)
        print("Running ML strategy with features:", features)
        strategy_best_results['strategy_svm_0'] = ('ML SVM n°0', my_params, execute_strategy(ml_svn_strategies_0.apply_svm_strategy, my_params))
        strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)
        del ml_svn_strategies_0
    # if 'strategy_svm_1' not in strategy_best_results:
    #     features = ['close', 'volume']
    #     ml_svn_strategies_1 = SVMstrategies(ml_btc_price_df, "1_" + ml_cache_prefix, features)
    #     ml_svn_strategies_1.maybe_train_model(ml_enable_train_cache)
    #     ml_svn_strategies_1.btc_price_df = btc_price_df # Use the same btc_price_df for the ML strategies as the basic strategies
    #     my_params = (100,)
    #     print("Running ML strategy with features:", features)
    #     strategy_best_results['strategy_svm_1'] = ('ML SVM n°1', my_params, execute_strategy(ml_svn_strategies_1.apply_svm_strategy, my_params))
    #     strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)
    #     del ml_svn_strategies_1

    if 'strategy_rf_0' not in strategy_best_results or True:
        features = ['close', 'volume']
        ml_rf_strategies_0 = RFstrategies(ml_btc_price_df, "0_" + ml_cache_prefix, features)
        ml_rf_strategies_0.maybe_train_model(False)
        ml_rf_strategies_0.btc_price_df = btc_price_df # Use the same btc_price_df for the ML strategies as the basic strategies
        my_params = (100,)
        print("Running ML RF strategy with features:", features)
        strategy_best_results['strategy_rf_0'] = ('ML RF n°0', my_params, execute_strategy(ml_rf_strategies_0.apply_rf_strategy, my_params))
        strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)
        del ml_rf_strategies_0

    if 'strategy_rf_1' not in strategy_best_results or True:
        features = ['close', 'volume', 'macd', 'close_pct_change']
        ml_rf_strategies_1 = RFstrategies(ml_btc_price_df, "1_" + ml_cache_prefix, features)
        ml_rf_strategies_1.maybe_train_model(False)
        ml_rf_strategies_1.btc_price_df = btc_price_df # Use the same btc_price_df for the ML strategies as the basic strategies
        my_params = (100,)
        print("Running ML RF strategy with features:", features)
        strategy_best_results['strategy_rf_1'] = ('ML RF n°1', my_params, execute_strategy(ml_rf_strategies_1.apply_rf_strategy, my_params))
        strategy_best_results, strategy_worst_results = store_results(strategy_best_results, strategy_worst_results)
        del ml_rf_strategies_1



    # order by strategy_best_results final_portfolio_value
    strategy_best_results = dict(sorted(strategy_best_results.items(), key=lambda item: item[1][2][0], reverse=False))
    # Print the best results
    for name, (description, params, (final_portfolio_value, _, _, _)) in strategy_best_results.items():
        print(f"{name}\tFinal portfolio value: {final_portfolio_value:.2f} USDT - Best parameters: {format_params(params)}")


if __name__ == "__main__":
    main()