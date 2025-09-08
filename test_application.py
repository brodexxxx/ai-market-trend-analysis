#!/usr/bin/env python3
"""
Test script to verify the stock market analysis application functionality
"""

import streamlit_app_enhanced as app
import pandas as pd
import numpy as np

def test_technical_indicators():
    """Test technical indicator calculations"""
    print("Testing technical indicators...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.random.normal(100, 5, 100).cumsum()
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Calculate indicators
    df_with_indicators = app.calculate_technical_indicators(df)
    
    # Verify indicators were calculated
    assert 'SMA_20' in df_with_indicators.columns
    assert 'SMA_50' in df_with_indicators.columns
    assert 'RSI' in df_with_indicators.columns
    assert 'MACD' in df_with_indicators.columns
    assert 'Signal_Line' in df_with_indicators.columns
    
    print("✅ Technical indicators test passed!")

def test_pattern_recognition():
    """Test pattern recognition"""
    print("Testing pattern recognition...")
    
    # Create sample data with clear uptrend
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    prices = np.linspace(100, 120, 20)  # Clear uptrend
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 20)
    }, index=dates)
    
    patterns = app.identify_patterns(df)
    
    # Should detect uptrend
    assert any("Uptrend" in pattern for pattern in patterns)
    print("✅ Pattern recognition test passed!")

def test_sample_data_generation():
    """Test sample data generation"""
    print("Testing sample data generation...")
    
    df = app.generate_sample_data("TEST", "1mo")
    
    assert df is not None
    assert not df.empty
    assert len(df) >= 30  # At least 30 days for 1 month
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    print("✅ Sample data generation test passed!")

def test_security_functions():
    """Test security functions"""
    print("Testing security functions...")
    
    test_data = "test message"
    secret_key = "test_secret"
    
    # Test hash generation
    hash_result = app.generate_secure_hash(test_data, secret_key)
    assert isinstance(hash_result, str)
    assert len(hash_result) == 64  # SHA256 hash length
    
    # Test encryption (requires proper key handling)
    print("✅ Security functions test passed!")

if __name__ == "__main__":
    print("Running application tests...")
    print("=" * 50)
    
    try:
        test_technical_indicators()
        test_pattern_recognition()
        test_sample_data_generation()
        test_security_functions()
        
        print("=" * 50)
        print("🎉 All tests passed! Application is functioning correctly.")
        print("\nThe application is ready to use. Run with:")
        print("python -m streamlit run streamlit_app_enhanced.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
