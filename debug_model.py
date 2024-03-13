import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
import seaborn as sns

ml_prefix = '0_2013-1-1-'
data_prefix = '2013-1-1-'

# Define the paths to your saved components
model_path = ml_prefix + 'svm_model.joblib'
scaler_path = ml_prefix + 'svm_scaler.joblib'
imputer_path = ml_prefix + 'svm_imputer.joblib'
btc_price_df_path = data_prefix + 'btc_price_df_cache.pkl'

# Load your saved components
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
imputer = joblib.load(imputer_path)
btc_price_df = joblib.load(btc_price_df_path)

# Slice the last third of the DataFrame
total_rows = len(btc_price_df)
btc_price_df = btc_price_df.iloc[-total_rows//6:]


#################################################################

# features = ['close', 'volume', 'macd', 'signal', 'rsi', 'sma_20', 'ema_20', 'close_pct_change']
# Best features are these:
features = ['close', 'volume', 'macd', 'close_pct_change']

# Calculate the percentage change and shift it up by one row
btc_price_df['Future_Price_Movement'] = btc_price_df['close'].pct_change().shift(-1)
# Since the last row now becomes NaN due to the shift, you can fill it with 0
btc_price_df['Future_Price_Movement'].fillna(0, inplace=True)
corr = btc_price_df[features + ['Future_Price_Movement']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# plt.savefig('correlation_matrix.png')


#################################################################

features = ['close', 'volume', 'macd', 'close_pct_change']

# Prepare the data
features_df = btc_price_df[features]
features_imputed = imputer.transform(features_df)
features_scaled = scaler.transform(features_imputed)

# Make predictions
predictions = svm_model.predict(features_scaled)

# Assuming you have a 'close' column for BTC closing prices
# btc_price_df['Predictions'] = np.append(predictions, 0)  # Append 0 for the last prediction to match the DataFrame length
btc_price_df['Predictions'] = predictions

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(btc_price_df['close'], label='BTC Close', alpha=0.5)

buy_signals = btc_price_df[btc_price_df['Predictions'] == 1].index
sell_signals = btc_price_df[btc_price_df['Predictions'] == 0].index

plt.scatter(btc_price_df.loc[buy_signals].index, btc_price_df.loc[buy_signals]['close'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(btc_price_df.loc[sell_signals].index, btc_price_df.loc[sell_signals]['close'], label='Sell Signal', marker='v', color='red', alpha=1)

plt.title('BTC Close Price and Trading Signals')
plt.xlabel('Date')
plt.ylabel('BTC Close Price')
plt.legend()


plt.figure(figsize=(6, 4))
counts, bins, patches = plt.hist(predictions, bins=np.arange(-0.5, 2.5), alpha=0.7, color='blue', edgecolor='black')
plt.xticks([0, 1], ['Sell', 'Buy'])
plt.title('Distribution of Predictions')
plt.xlabel('Prediction')
plt.ylabel('Count')

# Adding a legend manually
legend_labels = ['Sell', 'Buy']
plt.legend(handles=[Patch(color='blue', label=label) for label in legend_labels])


plt.show()
