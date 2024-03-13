import matplotlib.pyplot as plt
import joblib

prefix = '2022-1-1-'
# btc_price_df = joblib.load('btc_price_df_cache.pkl')
# strategy_best_results = joblib.load('strategy_best_results.pkl')
# strategy_worst_results = joblib.load('strategy_worst_results.pkl')

btc_price_df = joblib.load(prefix + 'btc_price_df_cache.pkl')
strategy_best_results = joblib.load(prefix + 'strategy_best_results.pkl')
strategy_worst_results = joblib.load(prefix + 'strategy_worst_results.pkl')

# order by strategy_best_results name
strategy_best_results = dict(sorted(strategy_best_results.items(), key=lambda item: item[0]))
strategy_worst_results = dict(sorted(strategy_worst_results.items(), key=lambda item: item[0]))

def format_tuple_floats(t):
    return tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, t))


# plot in a first figure all data related to btc_price_df
# Index(['open', 'high', 'low', 'close', 'volume', 'close_time',
#        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
#        'taker_buy_quote_asset_volume', 'ignore', 'macd', 'signal', 'histogram',
#        'rsi', 'sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200',
#        'close_pct_change'],
#       dtype='object')
fig, axs = plt.subplots(5, 1, figsize=(12, 10))

# Plot BTC/USDT and volume on the first subplot
axs[0].plot(btc_price_df['close'], label='BTC/USDT')
axs[0].plot(btc_price_df['volume'], label='Volume', color='green')
axs[0].legend()

# Plot the price change, MACD and RSI on the second subplot
axs[1].plot(btc_price_df['close_pct_change'], label='BTC/USDT % Change', color='orange')
axs[1].legend()

axs[2].plot(btc_price_df['rsi'], label='RSI', color='red')
axs[2].legend()

# Plot the moving averages on the third subplot
axs[3].plot(btc_price_df['sma_20'], label='SMA 20', color='orange')
axs[3].plot(btc_price_df['sma_50'], label='SMA 50', color='blue')
axs[3].plot(btc_price_df['sma_200'], label='SMA 200', color='red')
axs[3].legend()

axs[4].plot(btc_price_df['macd'], label='MACD', color='blue')
axs[4].plot(btc_price_df['signal'], label='Signal', color='red')
axs[4].plot(btc_price_df['histogram'], label='Histogram', color='green')
axs[4].legend()

# only save the figure without showing it
plt.savefig(prefix + 'plot_btc_price_data.png')
plt.close()

# Plot only the best strategies
plt.figure(figsize=(9, 6))

# Plot the portfolio value on the third subplot
for name, (description, params, (final_portfolio_value, portfolio_history_df, buy_signals, sell_signals)) in strategy_best_results.items():
    # plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label=f"{name} - {final_portfolio_value:.2f} USDT - {params}")

    plt.plot(portfolio_history_df, label=f"{name} - {final_portfolio_value:.2f} USDT - {description} - {format_tuple_floats(params)}")
    # Add buy and sell signals to the portfolio value plot
    # plt.scatter(buy_signals, portfolio_history_df.loc[buy_signals, 'value'], label=None, marker='^', color='blue')
    # plt.scatter(sell_signals, portfolio_history_df.loc[sell_signals, 'value'], label=None, marker='v', color='red')
plt.legend()
plt.title('Best Strategies')

# for name, (description, params, (final_portfolio_value, portfolio_history_df, buy_signals, sell_signals)) in strategy_worst_results.items():
#     # plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label=f"{name} - {final_portfolio_value:.2f} USDT - {params}")

#     axs[1].plot(portfolio_history_df, label=f"{name} - {final_portfolio_value:.2f} USDT - {description} - {format_tuple_floats(params)}")
#     # Add buy and sell signals to the portfolio value plot
#     axs[1].scatter(buy_signals, portfolio_history_df.loc[buy_signals, 'value'], label=None, marker='^', color='blue')
#     axs[1].scatter(sell_signals, portfolio_history_df.loc[sell_signals, 'value'], label=None, marker='v', color='red')
#     axs[1].legend()
#     axs[1].title.set_text('Worst Strategy')

plt.savefig(prefix + 'plot_results.png')
plt.tight_layout()



plt.show()