import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

def fetch_yahoo_finance_data(symbols, period='2y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance API
    """
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df['Symbol'] = symbol
            data[symbol] = df
            print(f"Fetched data for {symbol}: {len(df)} records")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return data

def clean_data(df):
    """
    Clean and preprocess the raw stock data
    """
    # Handle missing values
    df = df.dropna()
    
    # Ensure proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    return df

def calculate_basic_features(df):
    """
    Calculate basic financial features
    """
    df = df.copy()
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (rolling standard deviation)
    df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std()
    
    # Simple moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price relative to moving averages
    df['Price_vs_SMA20'] = df['Close'] / df['SMA_20']
    df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']
    
    return df

def create_target_variable(df, horizon=1):
    """
    Create target variable for prediction
    """
    df = df.copy()
    
    # Binary classification: 1 if price goes up, 0 if down
    df['Target_Direction'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    
    # Multi-class classification: Bullish, Bearish, Neutral
    threshold = 0.005  # 0.5% threshold for neutral zone
    future_return = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    
    df['Target_Class'] = pd.cut(future_return, 
                               bins=[-np.inf, -threshold, threshold, np.inf],
                               labels=['Bearish', 'Neutral', 'Bullish'])
    
    # Regression target: future return
    df['Target_Return'] = future_return
    
    return df

def preprocess_data(data_config):
    """
    Main preprocessing function
    """
    print("Starting data preprocessing...")
    
    # Fetch data from Yahoo Finance
    if data_config['yahoo_finance']['enabled']:
        symbols = data_config['yahoo_finance']['symbols']
        period = data_config['yahoo_finance']['period']
        interval = data_config['yahoo_finance']['interval']
        
        raw_data = fetch_yahoo_finance_data(symbols, period, interval)
    else:
        raise ValueError("No data source enabled in configuration")
    
    # Process each symbol's data
    processed_data = {}
    for symbol, df in raw_data.items():
        try:
            # Clean data and handle duplicates
            df_clean = clean_data(df)
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
            
            # Calculate basic features
            df_features = calculate_basic_features(df_clean)
            
            # Create target variables
            df_final = create_target_variable(df_features, horizon=1)
            
            processed_data[symbol] = df_final
            print(f"Processed data for {symbol}: {len(df_final)} records")
            
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
    
    # Combine all symbols into single DataFrame
    combined_df = pd.concat(processed_data.values(), axis=0)
    
    # Reset index to create unique indices
    combined_df = combined_df.reset_index()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    combined_df.to_csv('data/processed/processed_stock_data.csv')
    
    print(f"Data preprocessing completed. Total records: {len(combined_df)}")
    
    return combined_df

if __name__ == "__main__":
    # Test the preprocessing
    config = {
        'yahoo_finance': {
            'enabled': True,
            'symbols': ['AAPL', 'MSFT'],
            'period': '6mo',
            'interval': '1d'
        }
    }
    
    data = preprocess_data(config)
    print(data.head())
