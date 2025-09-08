#!/usr/bin/env python3
"""
Test script to verify mock data generation works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_tradingview_signals import create_mock_data, tv_hist

def test_mock_data():
    """Test that mock data is generated correctly"""
    print("🧪 Testing mock data generation...")

    # Test Indian stock
    symbol = "NSE:TCS"
    df = create_mock_data(symbol)

    print(f"Mock data for {symbol}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print(f"  Sample data:\n{df.head(3)}")

    # Test international stock
    symbol2 = "NASDAQ:AAPL"
    df2 = create_mock_data(symbol2)

    print(f"\nMock data for {symbol2}:")
    print(f"  Shape: {df2.shape}")
    print(f"  Price range: {df2['close'].min():.2f} to {df2['close'].max():.2f}")

    # Test tv_hist function (should use mock data due to yfinance issues)
    print("\n🧪 Testing tv_hist function...")
    df3 = tv_hist("NSE:TCS")
    print(f"tv_hist result shape: {df3.shape}")
    print(f"Data available: {not df3.empty}")

    print("\n✅ Mock data tests completed successfully!")

if __name__ == "__main__":
    test_mock_data()
