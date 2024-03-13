import matplotlib.pyplot as plt
import joblib

prefix = '2019-'
# btc_price_df = joblib.load('btc_price_df_cache.pkl')
# strategy_best_results = joblib.load('strategy_best_results.pkl')
# strategy_worst_results = joblib.load('strategy_worst_results.pkl')

btc_price_df = joblib.load(prefix + 'btc_price_df_cache.pkl')
strategy_best_results = joblib.load(prefix + 'strategy_best_results.pkl')
strategy_worst_results = joblib.load(prefix + 'strategy_worst_results.pkl')

# order by strategy_best_results name
strategy_best_results = dict(sorted(strategy_best_results.items(), key=lambda item: item[0]))
strategy_worst_results = dict(sorted(strategy_worst_results.items(), key=lambda item: item[0]))

# Plot all the data
fig, axs = plt.subplots(4, 1, figsize=(12, 10))

# Plot BTC/USDT on the first subplot
axs[0].plot(btc_price_df['close'], label='BTC/USDT')
axs[0].set_title('BTC/USDT Price')

# Plot the price change on the second subplot
axs[1].plot(btc_price_df['close'].pct_change(), label='BTC/USDT % Change', color='orange')
axs[1].title.set_text('BTC/USDT % Change')


def format_tuple_floats(t):
    return tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, t))

# Plot the portfolio value on the third subplot
for name, (description, params, (final_portfolio_value, portfolio_history_df, buy_signals, sell_signals)) in strategy_best_results.items():
    # plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label=f"{name} - {final_portfolio_value:.2f} USDT - {params}")

    axs[2].plot(portfolio_history_df, label=f"{name} - {final_portfolio_value:.2f} USDT - {description} - {format_tuple_floats(params)}")
    # Add buy and sell signals to the portfolio value plot
    axs[2].scatter(buy_signals, portfolio_history_df.loc[buy_signals, 'value'], label=None, marker='^', color='blue')
    axs[2].scatter(sell_signals, portfolio_history_df.loc[sell_signals, 'value'], label=None, marker='v', color='red')
    axs[2].legend()
    axs[2].title.set_text('Best Strategy')

for name, (description, params, (final_portfolio_value, portfolio_history_df, buy_signals, sell_signals)) in strategy_worst_results.items():
    # plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label=f"{name} - {final_portfolio_value:.2f} USDT - {params}")

    axs[3].plot(portfolio_history_df, label=f"{name} - {final_portfolio_value:.2f} USDT - {description} - {format_tuple_floats(params)}")
    # Add buy and sell signals to the portfolio value plot
    axs[3].scatter(buy_signals, portfolio_history_df.loc[buy_signals, 'value'], label=None, marker='^', color='blue')
    axs[3].scatter(sell_signals, portfolio_history_df.loc[sell_signals, 'value'], label=None, marker='v', color='red')
    axs[3].legend()
    axs[3].title.set_text('Worst Strategy')

plt.savefig(prefix + 'plot_results.png')
plt.tight_layout()
plt.show()