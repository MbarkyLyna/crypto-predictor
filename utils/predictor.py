import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_features(self, df):
        """Create technical indicators and features"""
        df = df.copy()
        
        # Price-based features
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_14'] = df['price'].rolling(window=14).mean()
        df['price_ma_30'] = df['price'].rolling(window=30).mean()
        
        # Volatility
        df['price_std_7'] = df['price'].rolling(window=7).std()
        
        # Returns
        df['return_1d'] = df['price'].pct_change(1)
        df['return_7d'] = df['price'].pct_change(7)
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Lag features
        for i in [1, 2, 3, 7]:
            df[f'price_lag_{i}'] = df['price'].shift(i)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df, target_col='price'):
        """Prepare features and target for training"""
        df = self.create_features(df)
        
        # Feature columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'price']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y, feature_cols
    
    def train(self, df):
        """Train the prediction model"""
        print("Training model...")
        
        X, y, feature_cols = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest (better for non-linear patterns)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        self.feature_cols = feature_cols
        
        # Calculate training accuracy
        predictions = self.model.predict(X_scaled)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        print(f"✓ Model trained. MAPE: {mape:.2f}%")
        
        return mape
    
    def predict_next_24h(self, df, hours=24):
        """Predict prices for next 24 hours"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Use last row with features
        df_with_features = self.create_features(df)
        last_row = df_with_features.iloc[-1]
        
        predictions = []
        current_price = last_row['price']
        
        # Simple approach: predict next step iteratively
        for hour in range(hours):
            # Prepare features from last known data
            features = last_row[self.feature_cols].values.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Predict
            predicted_price = self.model.predict(features_scaled)[0]
            
            # Store prediction
            predictions.append({
                'hour': hour + 1,
                'predicted_price': predicted_price,
                'timestamp': datetime.now() + timedelta(hours=hour+1)
            })
            
            # For next iteration (simplified - in reality would need full feature update)
            current_price = predicted_price
        
        return pd.DataFrame(predictions)
    
    def generate_signal(self, current_price, predicted_prices):
        """Generate BUY/SELL/HOLD signal"""
        avg_predicted = predicted_prices['predicted_price'].mean()
        change_pct = ((avg_predicted - current_price) / current_price) * 100
        
        if change_pct > 2:
            signal = "🟢 BUY"
            confidence = min(abs(change_pct) * 10, 100)
        elif change_pct < -2:
            signal = "🔴 SELL"
            confidence = min(abs(change_pct) * 10, 100)
        else:
            signal = "🟡 HOLD"
            confidence = 50
        
        return {
            'signal': signal,
            'confidence': confidence,
            'expected_change': change_pct,
            'current_price': current_price,
            'predicted_avg': avg_predicted
        }

# Test
if __name__ == "__main__":
    from data_fetcher import CryptoDataFetcher
    
    print("Fetching data...")
    fetcher = CryptoDataFetcher()
    df = fetcher.get_historical_data('bitcoin', days=90)
    
    print("Training predictor...")
    predictor = CryptoPredictor()
    mape = predictor.train(df)
    
    print("\nGenerating predictions...")
    predictions = predictor.predict_next_24h(df, hours=24)
    print(predictions.head())
    
    print("\nGenerating signal...")
    current_price = df['price'].iloc[-1]
    signal = predictor.generate_signal(current_price, predictions)
    print(signal)