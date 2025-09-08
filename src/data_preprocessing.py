import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import requests

def generate_sample_data(symbol, days=500):
    """
    Generate sample stock data for demonstration
    """
    np.random.seed(hash(symbol) % 2**32)  # Reproducible random seed
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk for price
    price_changes = np.random.normal(0.001, 0.02, days)  # Mean return 0.1%, std 2%
    prices = 100 * np.exp(np.cumsum(price_changes))  # Start at $100
    
    # Generate OHLC data
    highs = prices * (1 + np.random.uniform(0, 0.05, days))
    lows = prices * (1 - np.random.uniform(0, 0.05, days))
    opens = prices * (1 + np.random.normal(0, 0.01, days))
    volumes = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes,
        'Dividends': 0,
        'Stock Splits': 0,
        'Symbol': symbol
    }, index=dates)
    
    return df

def fetch_yahoo_finance_data(symbols, period='2y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance API, with fallback to sample data
    """
    # Create a session with headers to mimic browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period=period, interval=interval)
            if df.empty or len(df) == 0:
                raise ValueError("Empty data received")
            df['Symbol'] = symbol
            data[symbol] = df
            print(f"Fetched data for {symbol}: {len(df)} records")
        except Exception as e:
            print(f"Failed to get ticker '{symbol}' reason: {e}")
            print(f"Using sample data for {symbol}")
            df = generate_sample_data(symbol)
            data[symbol] = df
            print(f"Generated sample data for {symbol}: {len(df)} records")

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
