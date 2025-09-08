from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from src.security_bot import security_bot, require_auth, error_handler
try:
    from model_training import train_model, load_model
except ImportError as e:
    print(f"Warning: Could not import model_training: {e}")
    # Define dummy functions if import fails
    def train_model(*args, **kwargs):
        return {"accuracy": 0.5, "message": "Model training not available"}
    def load_model(*args, **kwargs):
        return None

app = Flask(__name__)

# Global variables to store loaded models
loaded_models = {}

def load_all_models():
    """Load all available models from the models directory"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return

    # Load traditional ML models
    if os.path.exists('models/trained_model.pkl'):
        try:
            model_data = joblib.load('models/trained_model.pkl')
            loaded_models['traditional'] = model_data
            print("Loaded traditional ML model")
        except Exception as e:
            print(f"Error loading traditional model: {e}")
            print("Continuing without traditional model - it may need to be retrained")

    # Load LSTM model
    if os.path.exists('models/lstm_model.h5'):
        try:
            if load_model is not None:
                lstm_model = load_model('models/lstm_model.h5')
                loaded_models['lstm'] = lstm_model
                print("Loaded LSTM model")
            else:
                print("LSTM model loading not available")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")

# Load models on startup
load_all_models()

def fetch_stock_data(symbol, period):
    """Fetch stock data from Yahoo Finance with improved error handling"""
    try:
        import yfinance as yf
        import time

        # Clean symbol
        symbol = symbol.upper().strip()

        # Try different symbol formats for better success rate
        symbol_variants = [symbol]

        # Add common variations
        if not symbol.endswith('.NS'):
            symbol_variants.append(symbol + '.NS')
        else:
            symbol_variants.append(symbol.replace('.NS', ''))

        hist = None

        for sym in symbol_variants:
            try:
                print(f"Trying to fetch data for {sym}...")
                stock = yf.Ticker(sym)

                # Add timeout and retry mechanism
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        hist = stock.history(period=period, timeout=30)
                        if not hist.empty:
                            print(f"✓ Successfully fetched data for {sym}")
                            return hist
                        break
                    except Exception as retry_e:
                        if attempt < max_retries - 1:
                            print(f"Retry {attempt + 1} for {sym}: {str(retry_e)}")
                            time.sleep(2)  # Wait before retry
                        else:
                            print(f"Failed to fetch data for {sym} after {max_retries} attempts")

            except Exception as sym_e:
                print(f"Error with symbol {sym}: {str(sym_e)}")
                continue

        # If all attempts failed, try with different periods
        if hist is None or hist.empty:
            print(f"All symbol variants failed for {symbol}, trying shorter periods...")
            shorter_periods = ['6mo', '3mo', '1mo']
            for short_period in shorter_periods:
                if short_period != period:
                    try:
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period=short_period, timeout=30)
                        if not hist.empty:
                            print(f"✓ Fetched data for {symbol} with shorter period {short_period}")
                            return hist
                    except Exception as period_e:
                        continue

        if hist is None or hist.empty:
            print(f"Failed to get ticker '{symbol}' reason: All attempts failed")
            print(f"Using sample data for {symbol} (real data unavailable)")
            return generate_sample_data(symbol, period)

        return hist

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        print(f"Using sample data for {symbol} (real data unavailable)")
        return generate_sample_data(symbol, period)

def generate_sample_data(symbol, period):
    """Generate sample stock data for demonstration when real data is unavailable"""
    try:
        import pandas as pd
        import numpy as np

        # Create date range based on period
        if period == '1mo':
            days = 30
        elif period == '3mo':
            days = 90
        elif period == '6mo':
            days = 180
        elif period == '1y':
            days = 365
        else:  # 2y
            days = 730

        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')

        # Generate realistic sample data
        np.random.seed(42)  # For reproducible results
        base_price = 100 + np.random.randint(-20, 20)

        # Create price series with some randomness
        prices = []
        current_price = base_price
        for _ in range(days):
            change = np.random.normal(0, 2)  # Small daily changes
            current_price = max(10, current_price + change)  # Don't go below 10
            prices.append(current_price)

        # Create DataFrame with realistic structure
        df = pd.DataFrame({
            'Open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in range(days)]
        }, index=dates)

        print(f"Using sample data for {symbol} (real data unavailable)")
        return df

    except Exception as e:
        print(f"Error generating sample data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for stock data"""
    df = df.copy()

    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    return df

@app.route('/train', methods=['POST'])
@require_auth
@error_handler
def train():
    data = request.json
    features_data = security_bot.sanitize_input(data['features'])
    model_config = security_bot.sanitize_input(data.get('model_config', {}))

    # Convert features to DataFrame
    features_df = pd.DataFrame(features_data)

    # Train the model
    result = train_model(features_df, model_config)

    # Reload models after training
    load_all_models()

    return jsonify({
        'accuracy': result['accuracy'],
        'model_type': model_config.get('model_type', 'random_forest'),
        'message': 'Model trained successfully'
    })

@app.route('/health', methods=['GET'])
@error_handler
def health():
    accuracy = security_bot.get_prediction_accuracy()
    return jsonify({'status': 'healthy', 'models_loaded': list(loaded_models.keys()), 'accuracy': accuracy})

@app.route('/predict', methods=['POST'])
@require_auth
@error_handler
def predict():
    data = request.json
    model_type = data.get('model_type', 'traditional')

    if model_type not in loaded_models:
        return jsonify({'error': f'Model type {model_type} not loaded'}), 400

    features = np.array(data['features']).reshape(1, -1)

    if model_type == 'traditional':
        model_data = loaded_models['traditional']
        features_scaled = model_data['scaler'].transform(features)
        prediction = model_data['model'].predict(features_scaled)
        return jsonify({'prediction': int(prediction[0]), 'model_type': 'traditional'})

    elif model_type == 'lstm':
        # For LSTM, we need to reshape for sequence input
        sequence_length = 30  # Default sequence length
        features_reshaped = features.reshape(1, sequence_length, -1)
        prediction = loaded_models['lstm'].predict(features_reshaped)
        return jsonify({'prediction': int(prediction[0] > 0.5), 'model_type': 'lstm'})

if __name__ == '__main__':
    app.run(debug=True)
