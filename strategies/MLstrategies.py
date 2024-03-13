import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
import os

# All available columns in btc_price_df are:
# Index(['open', 'high', 'low', 'close', 'volume', 'close_time',
#        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
#        'taker_buy_quote_asset_volume', 'ignore', 'macd', 'signal', 'histogram',
#        'rsi', 'sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200'],
#       dtype='object')

class MLstrategies:
    def __init__(self, btc_price_df, cache_prefix):
        self.btc_price_df = btc_price_df
        self.model_path = cache_prefix + 'svm_model.joblib'
        self.scaler_path = cache_prefix + 'scaler.joblib'
        self.imputer_path = cache_prefix + 'imputer.joblib'
        self.test_accuracy_path = cache_prefix + 'test_accuracy.joblib'

        # Attempt to load the model and scaler if they exist
        if os.path.exists(self.model_path):
            self.svm_model = joblib.load(self.model_path)
        else:
            self.svm_model = None
            
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = None

        if os.path.exists(self.imputer_path):
            self.imputer = joblib.load(self.imputer_path)
        else:
            self.imputer = None

        if os.path.exists(self.test_accuracy_path):
            self.test_accuracy = joblib.load(self.test_accuracy_path)
        else:
            self.test_accuracy = None

    def maybe_train_model(self, enable_cache=True):
        if self.svm_model is None or self.scaler is None or self.imputer is None or self.test_accuracy is None or not enable_cache:
            print("Model and scaler not found. Training the model...")
            self.preprocess_and_train()
        else:
            print("Model and scaler loaded from cache.")
        print(f"Test accuracy: {self.test_accuracy:.2f}")

    def optimize_hyperparameters(self, X_train, y_train, search_type='grid', n_iter=10):
        """
        Optimizes hyperparameters for the SVM model using either Grid Search or Random Search.

        :param X_train: Training feature data
        :param y_train: Training target data
        :param search_type: Type of search ('grid' for GridSearchCV or 'random' for RandomizedSearchCV)
        :param n_iter: Number of parameter settings sampled for RandomizedSearchCV. Ignored for GridSearchCV.
        """
        # Define the model
        svm = SVC()

        # Define the parameter grid/random search space
        param_distributions = {
            'C': [0.1, 1, 10, 100], 
            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        param_distributions = {
            'C': np.logspace(-2, 2, 5),
            'gamma': np.logspace(-4, 0, 5),
            # 'kernel': ['rbf', 'poly', 'sigmoid']
            'kernel': ['rbf', 'sigmoid']
        }
        # Choose between GridSearchCV and RandomizedSearchCV
        if search_type == 'grid':
            search = GridSearchCV(svm, param_distributions, cv=5, verbose=2, n_jobs=-1)
        else:
            search = RandomizedSearchCV(svm, param_distributions, n_iter=n_iter, cv=5, verbose=2, n_jobs=-1, random_state=42)

        # Perform the search
        search.fit(X_train, y_train)

        # Print the best parameters and return the best estimator
        print("Best parameters found:", search.best_params_)
        print("Best score:", search.best_score_)

        return search.best_estimator_

    def preprocess_and_train(self):
        # Feature selection
        features = self.btc_price_df[['close', 'volume', 'macd', 'signal', 'rsi', 'sma_20', 'ema_20', 'close_pct_change']]
        labels = np.where(self.btc_price_df['close'].shift(-1) > self.btc_price_df['close'], 1, 0)[:-1]
        features = features[:-1]  # Drop the last row to match labels' shape

        # Imputation of missing values
        self.imputer = SimpleImputer(strategy='mean')
        features_imputed = self.imputer.fit_transform(features)
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(features_imputed, labels, test_size=0.1, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scaler = scaler  # Save scaler for later use
        
        # SVM Model training
        # self.svm_model = SVC(kernel='rbf', C=1.0, probability=True)

        # self.svm_model = self.optimize_hyperparameters(X_train, y_train, search_type='grid')
        # Or, for Random Search: 
        self.svm_model = self.optimize_hyperparameters(X_train, y_train, search_type='random', n_iter=10)

        self.svm_model.fit(X_train_scaled, y_train)
        
        # test accuracy for evaluation
        self.test_accuracy = self.svm_model.score(X_test_scaled, y_test)

        # Cache the model and scaler
        joblib.dump(self.svm_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.imputer, self.imputer_path)
        joblib.dump(self.test_accuracy, self.test_accuracy_path)
        
        print("Model and scaler trained and cached.")

    def apply_svm_strategy(self, initial_owned_usdt):
        # Initialize variables
        current_usdt = initial_owned_usdt
        current_btc = 0
        portfolio_history = []
        buy_signals = []
        sell_signals = []
        
        # Iterate through btc_price_df to simulate the strategy
        for index, row in self.btc_price_df.iterrows():
            features = row[['close', 'volume', 'macd', 'signal', 'rsi', 'sma_20', 'ema_20', 'close_pct_change']].values.reshape(1, -1)
            features_df = pd.DataFrame(features, columns=['close', 'volume', 'macd', 'signal', 'rsi', 'sma_20', 'ema_20', 'close_pct_change'])
            features_imputed = self.imputer.transform(features_df)
            features_scaled = self.scaler.transform(features_imputed)
            
            # Predict the movement: 1 for buy, 0 for sell
            prediction = self.svm_model.predict(features_scaled)
            
            # Implementing the buy or sell logic based on prediction
            if prediction == 1 and current_usdt > 0:  # Buy signal
                current_btc = current_usdt / row['close']
                current_usdt = 0
                buy_signals.append(index)
            elif prediction == 0 and current_btc > 0:  # Sell signal
                current_usdt = current_btc * row['close']
                current_btc = 0
                sell_signals.append(index)
            
            portfolio_value = current_usdt + (current_btc * row['close'])
            portfolio_history.append((index, portfolio_value))
        
        # Convert portfolio history into DataFrame
        portfolio_history_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'value']).set_index('timestamp')
        final_portfolio_value = portfolio_history_df['value'].iloc[-1]
        return final_portfolio_value, portfolio_history_df, buy_signals, sell_signals
