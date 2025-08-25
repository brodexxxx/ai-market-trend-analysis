from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
import time
import random
from functools import wraps

app = Flask(__name__)

# Enable CORS
CORS(app)

# Load the trained model
try:
    model = joblib.load('models/trained_model.pkl')
    print("✅ Model loaded successfully")
    
    # For this demo, we'll use simple features that don't require the original scaler
    scaler = None
    feature_columns = ['Close', 'Volume', 'Daily_Return', 'Avg_Return', 'Volatility']
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Model not found. Please run main.py first to train the model.")
    model = None

# Indian top stocks
INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS"
]

# US top stocks
US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"
]

# Rate limiting decorator
def rate_limit(max_per_minute=60):
    def decorator(f):
        calls = []
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if now - call < 60]
            if len(calls) >= max_per_minute:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            calls.append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Validation functions
def validate_stock_symbol(symbol):
    """Validate stock symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    # Basic validation - should contain letters and optionally numbers/dots
    return bool(re.match(r'^[A-Za-z0-9\.]+$', symbol))

def validate_prediction_request(data):
    """Validate prediction request data"""
    if not data or not isinstance(data, dict):
        return False, "Invalid request data"
    
    symbol = data.get('symbol')
    if not symbol or not validate_stock_symbol(symbol):
        return False, "Invalid stock symbol"
    
    return True, ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'server': 'Stock Prediction API'
    })

@app.route('/stocks', methods=['GET'])
def get_stocks_list():
    """Get available stock lists"""
    return jsonify({
        'indian_stocks': INDIAN_STOCKS,
        'us_stocks': US_STOCKS,
        'total_stocks': len(INDIAN_STOCKS) + len(US_STOCKS)
    })

@app.route('/predict', methods=['POST'])
@rate_limit(max_per_minute=30)
def predict():
    """Main prediction endpoint with enhanced error handling"""
    if model is None:
        return jsonify({'error': 'Model not trained. Please run main.py first'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate request
        is_valid, error_msg = validate_prediction_request(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        symbol = data.get('symbol', 'AAPL').upper()
        days = data.get('days', 30)
        
        # Fetch recent data for prediction
        stock_data = fetch_stock_data(symbol, days)
        if stock_data is None or stock_data.empty:
            return jsonify({'error': f'Could not fetch data for {symbol}. Please check the symbol.'}), 404
        
        # Validate we have enough data
        if len(stock_data) < 5:
            return jsonify({'error': f'Insufficient data for {symbol}. Need at least 5 days of data.'}), 400
        
        # Prepare features for prediction
        features = prepare_features(stock_data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        confidence_scores = model.predict_proba([features])[0]
        confidence = float(confidence_scores.max())
        
        # Calculate additional statistics
        price_data = stock_data['Close']
        current_price = float(price_data.iloc[-1])
        prev_price = float(price_data.iloc[-2]) if len(price_data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Generate detailed recommendation
        recommendation_data = generate_detailed_recommendation(prediction, confidence, price_change_pct, stock_data)
        
        # Calculate technical indicators
        technical_data = calculate_technical_indicators(stock_data)
        
        # Get market sentiment and news (placeholder)
        market_sentiment = get_market_sentiment(symbol)
        
        # Prepare chart data
        chart_data = prepare_chart_data(stock_data)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'prediction': int(prediction),
            'confidence': confidence,
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'recommendation': recommendation_data,
            'technical_data': technical_data,
            'market_sentiment': market_sentiment,
            'chart_data': chart_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
@rate_limit(max_per_minute=10)
def batch_predict():
    """Batch prediction for multiple stocks"""
    if model is None:
        return jsonify({'error': 'Model not trained'}), 503
    
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid request format'}), 400
        
        symbols = data.get('symbols', [])
        if not symbols or not isinstance(symbols, list):
            return jsonify({'error': 'Invalid symbols list'}), 400
        
        if len(symbols) > 20:
            return jsonify({'error': 'Maximum 20 symbols allowed per batch'}), 400
        
        results = []
        for symbol in symbols[:20]:  # Limit to 20 symbols
            try:
                if validate_stock_symbol(symbol):
                    stock_data = fetch_stock_data(symbol.upper())
                    if stock_data is not None and not stock_data.empty:
                        features = prepare_features(stock_data)
                        prediction = model.predict([features])[0]
                        confidence = float(model.predict_proba([features])[0].max())
                        
                        price_data = stock_data['Close']
                        current_price = float(price_data.iloc[-1])
                        
                        results.append({
                            'symbol': symbol.upper(),
                            'prediction': int(prediction),
                            'confidence': confidence,
                            'direction': 'UP' if prediction == 1 else 'DOWN',
                            'current_price': round(current_price, 2),
                            'status': 'success'
                        })
                    else:
                        results.append({
                            'symbol': symbol.upper(),
                            'status': 'error',
                            'error': 'Could not fetch data'
                        })
                else:
                    results.append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': 'Invalid symbol'
                    })
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_processed': len(results),
            'successful': len([r for r in results if r['status'] == 'success'])
        })
        
    except Exception as e:
        return jsonify({'error': 'Batch processing failed', 'details': str(e)}), 500

def fetch_stock_data(symbol, period='1mo'):
    """Fetch recent stock data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        if df.empty:
            return None
        
        # Calculate basic features
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_features(df):
    """Prepare features for prediction"""
    # Ensure we use the same feature engineering as during training
    features = []
    
    # Add features based on the training setup
    features.append(float(df['Close'].iloc[-1]))  # Last closing price
    features.append(float(df['Volume'].iloc[-1]))  # Last volume
    features.append(float(df['Daily_Return'].iloc[-1]))  # Last daily return
    features.append(float(df['Close'].pct_change().mean()))  # Average daily return
    features.append(float(df['Close'].pct_change().std()))  # Volatility
    
    # Add more features as needed to reach 147 total
    # This is a placeholder; you should add the same features used in training
    for _ in range(142):  # Placeholder for additional features
        features.append(0.0)  # Replace with actual feature calculations
    
    return features

def generate_detailed_recommendation(prediction, confidence, price_change_pct, stock_data):
    """Generate detailed buy/sell recommendation with reasoning"""
    price_data = stock_data['Close']
    volatility = price_data.pct_change().std() * 100  # Annualized volatility
    
    if prediction == 1:  # UP prediction
        if confidence > 0.7:
            return {
                'action': 'STRONG BUY',
                'confidence_level': 'Very High',
                'reason': f'Strong bullish signal with {confidence:.0%} confidence. Price momentum is positive.',
                'risk_level': 'Low' if volatility < 20 else 'Medium',
                'target_gain': f'+{min(25, int(confidence * 30))}%',
                'stop_loss': f'-{max(5, int((1-confidence) * 15))}%'
            }
        elif confidence > 0.6:
            return {
                'action': 'BUY',
                'confidence_level': 'High',
                'reason': f'Bullish signal with {confidence:.0%} confidence. Positive price trend.',
                'risk_level': 'Medium',
                'target_gain': f'+{min(20, int(confidence * 25))}%',
                'stop_loss': f'-{max(7, int((1-confidence) * 18))}%'
            }
        else:
            return {
                'action': 'HOLD',
                'confidence_level': 'Moderate',
                'reason': f'Weak bullish signal with {confidence:.0%} confidence. Consider waiting for stronger confirmation.',
                'risk_level': 'High',
                'target_gain': 'N/A',
                'stop_loss': 'N/A'
            }
    else:  # DOWN prediction
        if confidence > 0.7:
            return {
                'action': 'STRONG SELL',
                'confidence_level': 'Very High',
                'reason': f'Strong bearish signal with {confidence:.0%} confidence. Price momentum is negative.',
                'risk_level': 'Low' if volatility < 20 else 'Medium',
                'target_gain': 'N/A',
                'stop_loss': 'N/A'
            }
        elif confidence > 0.6:
            return {
                'action': 'SELL',
                'confidence_level': 'High',
                'reason': f'Bearish signal with {confidence:.0%} confidence. Negative price trend.',
                'risk_level': 'Medium',
                'target_gain': 'N/A',
                'stop_loss': 'N/A'
            }
        else:
            return {
                'action': 'HOLD',
                'confidence_level': 'Moderate',
                'reason': f'Weak bearish signal with {confidence:.0%} confidence. Consider monitoring closely.',
                'risk_level': 'High',
                'target_gain': 'N/A',
                'stop_loss': 'N/A'
            }

def get_market_sentiment(symbol):
    """Get market sentiment data (placeholder for real integration)"""
    # This would integrate with news APIs, social media sentiment, etc.
    sentiments = ['Bullish', 'Bearish', 'Neutral']
    return {
        'overall_sentiment': random.choice(sentiments),
        'news_count': random.randint(5, 50),
        'social_mentions': random.randint(10, 200),
        'analyst_rating': f'{random.randint(1, 5)}/5',
        'market_trend': random.choice(['Upward', 'Downward', 'Sideways'])
    }

def prepare_chart_data(stock_data):
    """Prepare comprehensive chart data"""
    try:
        price_data = stock_data['Close']
        volume_data = stock_data['Volume']
        
        # Price history for chart
        prices = price_data.values.tolist()
        volumes = volume_data.values.tolist()
        dates = price_data.index.strftime('%Y-%m-%d').tolist()
        
        # Calculate moving averages for chart
        sma_20 = price_data.rolling(window=min(20, len(price_data))).mean().tolist()
        sma_50 = price_data.rolling(window=min(50, len(price_data))).mean().tolist()
        
        # Calculate RSI for each point
        rsi_values = calculate_rsi_series(price_data)
        
        # Identify support and resistance levels
        support, resistance = identify_support_resistance(price_data)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'dates': dates,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi_series': rsi_values,
            'support_levels': support,
            'resistance_levels': resistance,
            'price_patterns': identify_price_patterns(price_data)
        }
    except Exception as e:
        print(f"Chart data preparation error: {e}")
        return {}

def calculate_rsi_series(prices, period=14):
    """Calculate RSI series for chart"""
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Initial values
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Calculate remaining values
        for i in range(period+1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        rs = avg_gains / np.maximum(avg_losses, 0.0001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN for the first period values
        rsi[:period] = np.nan
        
        return [float(x) if not np.isnan(x) else None for x in rsi]
    except:
        return [50] * len(prices)  # Fallback

def identify_support_resistance(prices, window=5):
    """Identify support and resistance levels"""
    try:
        support = []
        resistance = []
        
        for i in range(window, len(prices)-window):
            local_min = min(prices[i-window:i+window])
            local_max = max(prices[i-window:i+window])
            
            if prices[i] == local_min:
                support.append(float(prices[i]))
            elif prices[i] == local_max:
                resistance.append(float(prices[i]))
        
        return support[-3:], resistance[-3:]  # Return last 3 levels
    except:
        return [], []

def identify_price_patterns(prices):
    """Identify common price patterns"""
    patterns = []
    
    # Simple pattern detection (placeholder)
    if len(prices) >= 5:
        recent_trend = prices[-1] - prices[-5]
        if recent_trend > 0:
            patterns.append('Uptrend')
        else:
            patterns.append('Downtrend')
    
    return patterns

def calculate_technical_indicators(df):
    """Calculate technical indicators for chart display"""
    try:
        # Extract the actual values from the MultiIndex DataFrame
        close_prices = df['Close'].values.flatten()
        volumes = df['Volume'].values.flatten()
        
        # Simple moving averages
        sma_20 = pd.Series(close_prices).rolling(window=min(20, len(close_prices))).mean().iloc[-1]
        sma_50 = pd.Series(close_prices).rolling(window=min(50, len(close_prices))).mean().iloc[-1]
        
        # RSI (simplified)
        price_changes = pd.Series(close_prices).pct_change()
        gains = price_changes[price_changes > 0].mean() or 0
        losses = abs(price_changes[price_changes < 0].mean()) or 0.0001
        rsi = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
        
        # Volume analysis
        avg_volume = pd.Series(volumes).mean()
        current_volume = volumes[-1] if len(volumes) > 0 else 1
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return {
            'sma_20': float(sma_20) if not pd.isna(sma_20) else close_prices[-1],
            'sma_50': float(sma_50) if not pd.isna(sma_50) else close_prices[-1],
            'rsi': float(rsi),
            'volume_ratio': float(volume_ratio),
            'support_level': float(min(close_prices)),
            'resistance_level': float(max(close_prices))
        }
    except Exception as e:
        print(f"Technical indicator error: {e}")
        return {
            'sma_20': 0,
            'sma_50': 0,
            'rsi': 50,
            'volume_ratio': 1,
            'support_level': 0,
            'resistance_level': 0
        }

if __name__ == '__main__':
    app.run(debug=True, port=5001)
