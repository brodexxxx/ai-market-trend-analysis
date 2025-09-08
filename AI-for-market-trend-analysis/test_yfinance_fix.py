#!/usr/bin/env python3
"""
Test script to diagnose yfinance issues with Indian stocks
"""

import yfinance as yf
import pandas as pd

def test_yfinance_formats():
    """Test different yfinance symbol formats for Indian stocks"""
    print("🧪 Testing yfinance symbol formats for Indian stocks...")

    test_symbols = ["TCS", "RELIANCE", "HDFCBANK"]
    formats = [".NS", ".BO", ""]  # NSE, BSE, no suffix

    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        for fmt in formats:
            yf_symbol = f"{symbol}{fmt}"
            try:
                stock = yf.Ticker(yf_symbol)
                df = stock.history(period="1mo")  # Test with 1 month data
                if not df.empty:
                    print(f"  ✅ {yf_symbol}: {len(df)} records")
                    print(f"     Latest price: {df['Close'].iloc[-1]:.2f}")
                else:
                    print(f"  ❌ {yf_symbol}: Empty data")
            except Exception as e:
                print(f"  ❌ {yf_symbol}: {str(e)}")

def test_alternative_approach():
    """Test alternative approach for Indian stocks"""
    print("\n🧪 Testing alternative approaches...")

    # Try direct ticker without suffix
    try:
        stock = yf.Ticker("TCS")
        df = stock.history(period="1mo")
        if not df.empty:
            print("✅ TCS (no suffix): Success")
            print(f"   Latest price: {df['Close'].iloc[-1]:.2f}")
        else:
            print("❌ TCS (no suffix): Empty data")
    except Exception as e:
        print(f"❌ TCS (no suffix): {str(e)}")

    # Try with different periods
    try:
        stock = yf.Ticker("TCS.NS")
        df = stock.history(period="1wk")  # Try 1 week
        if not df.empty:
            print("✅ TCS.NS (1wk): Success")
            print(f"   Latest price: {df['Close'].iloc[-1]:.2f}")
        else:
            print("❌ TCS.NS (1wk): Empty data")
    except Exception as e:
        print(f"❌ TCS.NS (1wk): {str(e)}")

if __name__ == "__main__":
    test_yfinance_formats()
    test_alternative_approach()
