import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class RFstrategies:
    def __init__(self, btc_price_df, cache_prefix, features):
        self.btc_price_df = btc_price_df
        self.model_path = cache_prefix + 'rf_model.joblib'
        self.scaler_path = cache_prefix + 'rf_scaler.joblib'
        self.imputer_path = cache_prefix + 'rf_imputer.joblib'
        self.test_accuracy_path = cache_prefix + 'rf_test_accuracy.joblib'
        self.features = features

        if os.path.exists(self.model_path):
            self.rf_model = joblib.load(self.model_path)
        else:
            self.rf_model = None
            
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
        if self.rf_model is None or self.scaler is None or self.imputer is None or self.test_accuracy is None or not enable_cache:
            print("Model and scaler not found. Training the model...")
            self.preprocess_and_train()
        else:
            print("Model and scaler loaded from cache.")
        print(f"Test accuracy: {self.test_accuracy:.2f}")

    def optimize_hyperparameters(self, X_train, y_train, search_type='grid', n_iter=10):
        rf = RandomForestClassifier()

        param_distributions = {
            'n_estimators': [100, 200, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        if search_type == 'grid':
            search = GridSearchCV(rf, param_distributions, cv=5, verbose=2, n_jobs=-1)
        else:
            search = RandomizedSearchCV(rf, param_distributions, n_iter=n_iter, cv=5, verbose=2, n_jobs=-1, random_state=42)

        search.fit(X_train, y_train)

        print("Best parameters found:", search.best_params_)
        print("Best score:", search.best_score_)

        return search.best_estimator_

    def preprocess_and_train(self):
        features = self.btc_price_df[self.features]
        labels = np.where(self.btc_price_df['close'].shift(-1) > self.btc_price_df['close'], 1, 0)[:-1]
        features = features[:-1]

        self.imputer = SimpleImputer(strategy='mean')
        features_imputed = self.imputer.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(features_imputed, labels, test_size=0.1, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scaler = scaler
        
        self.rf_model = self.optimize_hyperparameters(X_train, y_train, search_type='random', n_iter=5)

        self.rf_model.fit(X_train_scaled, y_train)
        
        self.test_accuracy = self.rf_model.score(X_test_scaled, y_test)

        joblib.dump(self.rf_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.imputer, self.imputer_path)
        joblib.dump(self.test_accuracy, self.test_accuracy_path)
        
        print("Model and scaler trained and cached.")


    def apply_rf_strategy(self, initial_owned_usdt):
        # Initialize variables
        current_usdt = initial_owned_usdt
        current_btc = 0
        portfolio_history = []
        buy_signals = []
        sell_signals = []
        
        # Iterate through btc_price_df to simulate the strategy
        for index, row in self.btc_price_df.iterrows():
            features = row[self.features].values.reshape(1, -1)
            features_df = pd.DataFrame(features, columns=self.features)
            features_imputed = self.imputer.transform(features_df)
            features_scaled = self.scaler.transform(features_imputed)
            
            # Predict the movement: 1 for buy, 0 for sell
            prediction = self.rf_model.predict(features_scaled)
            
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
