import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app_enhanced import (
    generate_ai_stock_recommendation,
    generate_long_term_recommendation,
    calculate_technical_indicators,
    identify_patterns,
    generate_ai_response,
    detect_system_issues,
    check_data_sources
)
import pandas as pd
import numpy as np

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
    print(f"Long-Term Recommendation length: {len(recommendation)} characters")
    
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

def test_ai_response():
    """Test AI response generation"""
    print("\nTesting AI Response Function...")
    
    test_prompts = [
        "Hello",
        "What do you think about Apple stock?",
        "How's the market today?",
        "Can you diagnose my system?",
        "What's your opinion on risk management?"
    ]
    
    for prompt in test_prompts:
        response = generate_ai_response(prompt)
        print(f"Prompt: '{prompt}' -> Response length: {len(response)}")
    
    return True

def test_system_diagnostics():
    """Test system diagnostics functions"""
    print("\nTesting System Diagnostics Functions...")
    
    # Test system issues detection
    issues = detect_system_issues()
    print(f"System issues detected: {issues}")
    
    # Test data sources check
    data_status = check_data_sources()
    print(f"Data sources status: {data_status}")
    
    return issues is not None and data_status is not None

def main():
    """Run all tests"""
    print("Running comprehensive function tests...")
    print("=" * 50)
    
    tests = [
        test_ai_stock_recommendation,
        test_long_term_recommendation,
        test_technical_indicators,
        test_pattern_recognition,
        test_ai_response,
        test_system_diagnostics
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
    sys.exit(0 if success else 1)
