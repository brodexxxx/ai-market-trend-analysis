import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna
import warnings
import os
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df, config):
    """
    Calculate comprehensive technical indicators
    """
    df = df.copy()
    
    # Process each symbol separately to avoid issues with mixed data
    symbols = df['Symbol'].unique()
    results = []
    
    for symbol in symbols:
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        # Ensure we have a clean copy without NaN values for technical analysis
        # Only drop rows where essential columns have NaN values
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        symbol_clean = symbol_data.dropna(subset=essential_cols)
        
        if len(symbol_clean) < 20:  # Minimum records needed for most indicators
            print(f"Warning: Not enough data for {symbol} after cleaning ({len(symbol_clean)} records)")
            continue
        
        try:
            # Add all technical indicators from ta library
            df_ta = add_all_ta_features(
                symbol_clean,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                fillna=True
            )
            
            # Additional custom technical indicators
            df_ta = calculate_custom_indicators(df_ta, config)
            
            results.append(df_ta)
            
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {e}")
            # Fallback: use basic indicators only
            try:
                df_basic = calculate_custom_indicators(symbol_clean, config)
                results.append(df_basic)
            except Exception as e2:
                print(f"Even basic indicators failed for {symbol}: {e2}")
    
    if not results:
        raise ValueError("No valid data after technical indicator calculation")
    
    # Combine all symbols
    final_df = pd.concat(results, axis=0)
    
    return final_df

def calculate_custom_indicators(df, config):
    """
    Calculate custom technical indicators
    """
    df = df.copy()
    
    # Moving Averages
    for period in config.get('sma_periods', [5, 10, 20, 50, 200]):
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    for period in config.get('ema_periods', [12, 26]):
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # RSI
    rsi_period = config.get('rsi_period', 14)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()
    
    # MACD
    macd_fast = config.get('macd_fast', 12)
    macd_slow = config.get('macd_slow', 26)
    macd_signal = config.get('macd_signal', 9)
    
    macd = ta.trend.MACD(df['Close'], window_fast=macd_fast, 
                        window_slow=macd_slow, window_sign=macd_signal)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb_period = config.get('bollinger_period', 20)
    bb_std = config.get('bollinger_std', 2)
    
    bb = ta.volatility.BollingerBands(df['Close'], window=bb_period, window_dev=bb_std)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    # Average True Range (ATR)
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # On-Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    
    return df

def calculate_statistical_features(df, config):
    """
    Calculate statistical features
    """
    df = df.copy()
    
    # Volatility features
    volatility_window = config.get('volatility_window', 20)
    df['Volatility'] = df['Close'].pct_change().rolling(window=volatility_window).std()
    
    # Momentum features
    momentum_window = config.get('momentum_window', 10)
    df['Momentum'] = df['Close'] / df['Close'].shift(momentum_window) - 1
    
    # Return features
    return_window = config.get('return_window', 1)
    for window in [1, 5, 10, 20]:
        df[f'Return_{window}D'] = df['Close'].pct_change(window)
    
    # Price change features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Volume features
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    return df

def create_lag_features(df, max_lag=5):
    """
    Create lagged features for time series analysis
    """
    df = df.copy()
    
    # Lagged price features
    for lag in range(1, max_lag + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
    
    # Rolling statistics
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
    df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the feature matrix
    """
    # Drop rows with too many missing values
    df_clean = df.dropna(thresh=len(df.columns) * 0.8)
    
    # Fill remaining missing values with forward fill then backward fill
    df_clean = df_clean.ffill().bfill()
    
    # Drop any remaining rows with missing values
    df_clean = df_clean.dropna()
    
    return df_clean

def engineer_features(data, feature_config):
    """
    Main feature engineering function
    """
    print("Starting feature engineering...")
    
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(data, feature_config['technical_indicators'])
    
    # Calculate statistical features
    df_with_stats = calculate_statistical_features(df_with_indicators, feature_config['statistical_features'])
    
    # Create lag features
    df_with_lags = create_lag_features(df_with_stats, max_lag=5)
    
    # Handle missing values
    final_features = handle_missing_values(df_with_lags)
    
    # Save engineered features
    os.makedirs('data/features', exist_ok=True)
    final_features.to_csv('data/features/engineered_features.csv')
    
    print(f"Feature engineering completed. Final shape: {final_features.shape}")
    
    return final_features

if __name__ == "__main__":
    # Test the feature engineering
    import yfinance as yf
    
    # Fetch sample data
    df = yf.download('AAPL', period='6mo', interval='1d')
    df['Symbol'] = 'AAPL'
    
    # Test configuration
    config = {
        'technical_indicators': {
            'sma_periods': [5, 10, 20],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2
        },
        'statistical_features': {
            'volatility_window': 20,
            'momentum_window': 10,
            'return_window': 1
        }
    }
    
    features = engineer_features(df, config)
    print(features.head())
    print(f"Number of features: {len(features.columns)}")
