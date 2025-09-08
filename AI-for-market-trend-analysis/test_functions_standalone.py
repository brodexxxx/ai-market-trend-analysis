import pandas as pd
import numpy as np

# Replicate the functions from streamlit_app_enhanced.py for testing
def generate_ai_stock_recommendation(df, symbol):
    """Generate AI-powered stock recommendation based on technical analysis"""
    if df is None or df.empty:
        return "⚠️ Insufficient data for AI recommendation"
    
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    
    # AI analysis logic
    bullish_signals = 0
    bearish_signals = 0
    
    # Price vs Moving Averages
    if current_price > sma_20:
        bullish_signals += 1
    else:
        bearish_signals += 1
        
    if current_price > sma_50:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # RSI analysis
    if rsi < 30:
        bullish_signals += 2  # Strong oversold signal
    elif rsi > 70:
        bearish_signals += 2  # Strong overbought signal
    elif 30 <= rsi <= 50:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # MACD analysis
    if macd > signal:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Volume analysis
    avg_volume = df['Volume'].mean()
    recent_volume = df['Volume'].iloc[-5:].mean()
    if recent_volume > avg_volume * 1.2:
        bullish_signals += 1
    
    # Generate recommendation
    if bullish_signals - bearish_signals >= 3:
        return f"🎯 **STRONG BUY** - {symbol} shows multiple bullish signals for potential growth"
    elif bullish_signals - bearish_signals >= 1:
        return f"👍 **BUY** - {symbol} demonstrates favorable conditions for investment"
    elif bullish_signals == bearish_signals:
        return f"🤔 **HOLD** - {symbol} shows mixed signals, consider waiting"
    else:
        return f"⏸️ **WAIT** - {symbol} shows bearish tendencies, wait for better entry"

def generate_long_term_recommendation(df, symbol):
    """Generate long-term investment recommendation based on technical analysis"""
    if df is None or df.empty:
        return "⚠️ Insufficient data for recommendation"
    
    # Calculate long-term indicators
    current_price = df['Close'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
    rsi = df['RSI'].iloc[-1]
    
    recommendation = f"**Analysis for {symbol}:**\n\n"
    
    # Price vs Moving Averages
    if sma_200 is not None:
        if current_price > sma_200:
            recommendation += "✅ **Price above 200-day SMA** - Bullish long-term trend\n"
        else:
            recommendation += "⚠️ **Price below 200-day SMA** - Bearish long-term trend\n"
    
    if current_price > sma_50:
        recommendation += "✅ **Price above 50-day SMA** - Medium-term bullish\n"
    else:
        recommendation += "⚠️ **Price below 50-day SMA** - Medium-term bearish\n"
    
    # RSI analysis
    if rsi < 30:
        recommendation += "📈 **Oversold (RSI < 30)** - Potential buying opportunity\n"
    elif rsi > 70:
        recommendation += "📉 **Overbought (RSI > 70)** - Consider taking profits\n"
    else:
        recommendation += "📊 **RSI in neutral range** - Monitor for entry points\n"
    
    # Volume analysis
    avg_volume = df['Volume'].mean()
    recent_volume = df['Volume'].iloc[-5:].mean()
    if recent_volume > avg_volume * 1.5:
        recommendation += "📊 **High volume activity** - Increased investor interest\n"
    
    # Final recommendation
    bullish_signals = sum([
        current_price > sma_50,
        sma_200 is not None and current_price > sma_200,
        rsi < 40,
        recent_volume > avg_volume * 1.2
    ])
    
    if bullish_signals >= 3:
        recommendation += "\n🎯 **STRONG BUY** - Multiple bullish indicators align for long-term growth"
    elif bullish_signals >= 2:
        recommendation += "\n👍 **BUY** - Favorable conditions for long-term investment"
    elif bullish_signals >= 1:
        recommendation += "\n🤔 **HOLD** - Monitor for better entry points"
    else:
        recommendation += "\n⏸️ **WAIT** - Consider waiting for more favorable conditions"
    
    return recommendation

def calculate_technical_indicators(df):
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

def identify_patterns(df):
    patterns = []
    prices = df['Close'].values
    
    # Simple pattern detection
    if len(prices) >= 5:
        # Check for uptrend
        if prices[-1] > prices[-5]:
            patterns.append("📈 Uptrend")
        
        # Check for downtrend
        if prices[-1] < prices[-5]:
            patterns.append("📉 Downtrend")
        
        # Check for consolidation
        recent_range = max(prices[-10:]) - min(prices[-10:])
        if recent_range < (df['Close'].std() * 0.5):
            patterns.append("🔄 Consolidation")
    
    return patterns

def test_ai_stock_recommendation():
    """Test the AI stock recommendation function"""
    print("Testing AI Stock Recommendation Function...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 2, 100))
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Test recommendation
    recommendation = generate_ai_stock_recommendation(df, 'AAPL')
    print(f"AI Recommendation: {recommendation}")
    
    return recommendation is not None and len(recommendation) > 0

def test_long_term_recommendation():
    """Test the long-term recommendation function"""
    print("\nTesting Long-Term Recommendation Function...")
    
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1.5, 500))
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Test recommendation
    recommendation = generate_long_term_recommendation(df, 'AAPL')
    print(f"Long-Term Recommendation: {recommendation[:200]}...")  # Show first 200 chars
    
    return recommendation is not None and len(recommendation) > 0

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\nTesting Technical Indicators Function...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 2, 50))
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Check if indicators were added
    indicators_present = all(indicator in df_with_indicators.columns 
                          for indicator in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line'])
    
    print(f"Indicators calculated: {indicators_present}")
    print(f"DataFrame shape: {df_with_indicators.shape}")
    print(f"Sample RSI values: {df_with_indicators['RSI'].dropna().head(3).tolist()}")
    
    return indicators_present

def test_pattern_recognition():
    """Test pattern recognition function"""
    print("\nTesting Pattern Recognition Function...")
    
    # Create sample data with clear patterns
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    
    # Create uptrend data
    uptrend_prices = np.linspace(100, 120, 20)
    
    df_uptrend = pd.DataFrame({
        'Open': uptrend_prices * 0.99,
        'High': uptrend_prices * 1.01,
        'Low': uptrend_prices * 0.98,
        'Close': uptrend_prices,
        'Volume': np.random.randint(1000000, 3000000, 20)
    }, index=dates)
    
    patterns = identify_patterns(df_uptrend)
    print(f"Detected patterns: {patterns}")
    
    return patterns is not None

def main():
    """Run all tests"""
    print("Running comprehensive function tests...")
    print("=" * 50)
    
    tests = [
        test_ai_stock_recommendation,
        test_long_term_recommendation,
        test_technical_indicators,
        test_pattern_recognition,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"Test {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"Test FAILED with error: {e}")
            results.append(False)
        print("-" * 30)
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✅ All functions are working properly!")
        return True
    else:
        print("❌ Some functions need attention")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
