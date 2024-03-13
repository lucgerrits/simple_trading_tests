import pandas as pd


class Strategies:
    def __init__(self, btc_price_df):
        self.btc_price_df = btc_price_df

    def apply_strategy_just_hold(self, nothing, initial_owned_usdt):
        # print(f"Applying strategy with initial_owned_usdt={initial_owned_usdt}")
        # just hold the initial amount of BTC that can be bought with the initial USDT
        initial_btc_price = self.btc_price_df['close'].iloc[0]
        initial_btc_quantity = initial_owned_usdt / initial_btc_price
        final_portfolio_value = initial_btc_quantity * self.btc_price_df['close'].iloc[-1]
        portfolio_history = []
        buy_signals = []
        buy_signals.append(self.btc_price_df.index[0])

        for timestamp, current_price in self.btc_price_df['close'].items():
            portfolio_history.append((timestamp, initial_btc_quantity * current_price))
        portfolio_history_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'value']).set_index('timestamp')
        final_portfolio_value = portfolio_history_df['value'].iloc[-1]
        return final_portfolio_value, portfolio_history_df, buy_signals, []

    def apply_strategy(self, sell_percentage, buy_percentage, initial_owned_usdt):
        # print(f"Applying strategy with sell_percentage={sell_percentage}, buy_percentage={buy_percentage}, initial_owned_usdt={initial_owned_usdt}")
        previous_price = self.btc_price_df['close'].iloc[0]
        # Initialize variables to track the buy and sell signals
        last_action = "HOLD" # Start with HOLD, can be BUY or SELL
        quantity_owned_btc = 0 # Assuming starting with 1 BTC for the initial price of previous_price
        quantity_owned_usdt = initial_owned_usdt  # Assuming starting with 0 USDT for simplicity
        portfolio_history = []
        buy_signals = []
        sell_signals = []

        for timestamp, current_price in self.btc_price_df['close'].items():
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

    def apply_strategy_with_volume(self, sell_percentage, buy_percentage, initial_owned_usdt, volume_factor):
        # print(f"Applying strategy with sell_percentage={sell_percentage}, buy_percentage={buy_percentage}, initial_owned_usdt={initial_owned_usdt}, volume_factor={volume_factor}")
        previous_price = self.btc_price_df['close'].iloc[0]
        previous_volume = self.btc_price_df['volume'].iloc[0]
        last_action = "HOLD"
        quantity_owned_btc = 0
        quantity_owned_usdt = initial_owned_usdt
        portfolio_history = []
        buy_signals = []
        sell_signals = []

        for timestamp, row in self.btc_price_df.iterrows():
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

    def apply_strategy_with_volume_and_macd(self, sell_percentage, buy_percentage, initial_owned_usdt, volume_factor, macd_fast, macd_slow, macd_signal):
        previous_price = self.btc_price_df['close'].iloc[0]
        previous_volume = self.btc_price_df['volume'].iloc[0]
        last_action = "HOLD"
        quantity_owned_btc = 0
        quantity_owned_usdt = initial_owned_usdt
        portfolio_history = []
        buy_signals = []
        sell_signals = []

        # Calculate MACD components
        exp1 = self.btc_price_df['close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = self.btc_price_df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=macd_signal, adjust=False).mean()

        for timestamp, row in self.btc_price_df.iterrows():
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
