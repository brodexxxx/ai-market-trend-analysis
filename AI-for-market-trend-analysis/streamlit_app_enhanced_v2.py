import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import hashlib
import hmac
import base64
import os
from cryptography.fernet import Fernet

# Security functions
def generate_secure_hash(data, secret_key):
    """Generate HMAC SHA256 hash for data integrity"""
    if isinstance(data, str):
        data = data.encode()
    return hmac.new(secret_key.encode(), data, hashlib.sha256).hexdigest()

def encrypt_data(data, key):
    """Encrypt sensitive data using Fernet encryption"""
    if isinstance(data, str):
        data = data.encode()
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_data(encrypted_data, key):
    """Decrypt data using Fernet encryption"""
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()

# Generate a secure key (in production, this should be stored securely)
SECRET_KEY = Fernet.generate_key().decode()

# Enhanced long-term investment recommendation function
def generate_long_term_recommendation(df, symbol):
    """Generate detailed long-term investment recommendation with reasoning"""
    if df is None or df.empty:
        return "⚠️ Insufficient data for recommendation"
    
    # Calculate long-term indicators
    current_price = df['Close'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal_line = df['Signal_Line'].iloc[-1]
    
    # Volume analysis
    avg_volume = df['Volume'].mean()
    recent_volume = df['Volume'].iloc[-5:].mean()
    volume_ratio = recent_volume / avg_volume
    
    # Price momentum
    price_change_1m = (current_price / df['Close'].iloc[-30] - 1) * 100 if len(df) >= 30 else 0
    price_change_3m = (current_price / df['Close'].iloc[-90] - 1) * 100 if len(df) >= 90 else 0
    
    recommendation = f"## 📊 Detailed Analysis for {symbol}\n\n"
    
    # Technical Analysis Section
    recommendation += "### 🔍 Technical Indicators:\n\n"
    
    # Price vs Moving Averages
    if sma_200 is not None:
        if current_price > sma_200:
            recommendation += "✅ **Price above 200-day SMA** - Strong long-term bullish trend\n"
            recommendation += f"   - Current: ${current_price:.2f} vs 200-SMA: ${sma_200:.2f}\n"
            recommendation += "   - **Why this matters:** Price staying above 200-day SMA indicates sustained upward momentum\n"
        else:
            recommendation += "⚠️ **Price below 200-day SMA** - Long-term bearish pressure\n"
            recommendation += f"   - Current: ${current_price:.2f} vs 200-SMA: ${sma_200:.2f}\n"
            recommendation += "   - **Why this matters:** May indicate longer-term weakness in the stock\n"
    
    if current_price > sma_50:
        recommendation += "✅ **Price above 50-day SMA** - Medium-term bullish momentum\n"
        recommendation += f"   - Current: ${current_price:.2f} vs 50-SMA: ${sma_50:.2f}\n"
        recommendation += "   - **Why this matters:** Shows recent positive price action\n"
    else:
        recommendation += "⚠️ **Price below 50-day SMA** - Medium-term bearish pressure\n"
        recommendation += f"   - Current: ${current_price:.2f} vs 50-SMA: ${sma_50:.2f}\n"
        recommendation += "   - **Why this matters:** Suggests recent selling pressure\n"
    
    # RSI analysis with detailed reasoning
    recommendation += "\n### 📈 RSI Analysis:\n\n"
    if rsi < 30:
        recommendation += "📈 **Oversold (RSI < 30)** - Strong buying opportunity\n"
        recommendation += f"   - RSI Value: {rsi:.1f}\n"
        recommendation += "   - **Why this matters:** Historically, stocks often bounce back from oversold conditions\n"
        recommendation += "   - **Action:** Consider accumulating at these levels for potential rebound\n"
    elif rsi > 70:
        recommendation += "📉 **Overbought (RSI > 70)** - Consider profit-taking\n"
        recommendation += f"   - RSI Value: {rsi:.1f}\n"
        recommendation += "   - **Why this matters:** High RSI suggests potential pullback risk\n"
        recommendation += "   - **Action:** Partial profit-taking may be prudent\n"
    else:
        recommendation += "📊 **RSI in neutral range** - Monitor for entry points\n"
        recommendation += f"   - RSI Value: {rsi:.1f}\n"
        recommendation += "   - **Why this matters:** Stock is neither overbought nor oversold\n"
        recommendation += "   - **Action:** Wait for clearer signals or better entry points\n"
    
    # MACD analysis
    recommendation += "\n### 🔄 MACD Analysis:\n\n"
    if macd > signal_line:
        recommendation += "✅ **MACD above Signal Line** - Bullish momentum\n"
        recommendation += f"   - MACD: {macd:.4f}, Signal: {signal_line:.4f}\n"
        recommendation += "   - **Why this matters:** Positive momentum indicator\n"
    else:
        recommendation += "⚠️ **MACD below Signal Line** - Bearish momentum\n"
        recommendation += f"   - MACD: {macd:.4f}, Signal: {signal_line:.4f}\n"
        recommendation += "   - **Why this matters:** Negative momentum indicator\n"
    
    # Volume analysis
    recommendation += "\n### 📊 Volume Analysis:\n\n"
    if volume_ratio > 1.5:
        recommendation += "🔥 **High volume activity** - Strong investor interest\n"
        recommendation += f"   - Recent volume: {recent_ratio:.1f}x average\n"
        recommendation += "   - **Why this matters:** High volume often precedes significant price moves\n"
    elif volume_ratio > 1.2:
        recommendation += "📈 **Above average volume** - Increased activity\n"
        recommendation += f"   - Recent volume: {volume_ratio:.1f}x average\n"
        recommendation += "   - **Why this matters:** Suggests growing investor attention\n"
    else:
        recommendation += "📉 **Below average volume** - Low activity\n"
        recommendation += f"   - Recent volume: {volume_ratio:.1f}x average\n"
        recommendation += "   - **Why this matters:** May indicate lack of conviction in current price levels\n"
    
    # Price momentum
    recommendation += "\n### 🚀 Price Momentum:\n\n"
    if price_change_1m > 0:
        recommendation += f"📈 **1-month performance: +{price_change_1m:.1f}%** - Positive short-term momentum\n"
    else:
        recommendation += f"📉 **1-month performance: {price_change_1m:.1f}%** - Negative short-term momentum\n"
    
    if price_change_3m > 0:
        recommendation += f"📈 **3-month performance: +{price_change_3m:.1f}%** - Positive medium-term momentum\n"
    else:
        recommendation += f"📉 **3-month performance: {price_change_3m:.1f}%** - Negative medium-term momentum\n"
    
    # Final recommendation with detailed reasoning
    recommendation += "\n## 🎯 Investment Recommendation:\n\n"
    
    bullish_signals = sum([
        current_price > sma_50,
        sma_200 is not None and current_price > sma_200,
        rsi < 40,
        volume_ratio > 1.2,
        macd > signal_line,
        price_change_1m > 0
    ])
    
    if bullish_signals >= 5:
        recommendation += "🔥 **STRONG BUY RECOMMENDATION**\n\n"
        recommendation += "**Why BUY now:**\n"
        recommendation += "- Multiple bullish technical indicators align\n"
        recommendation += "- Strong price momentum across timeframes\n"
        recommendation += "- High volume suggests institutional interest\n"
        recommendation += "- Oversold conditions provide good entry point\n"
        recommendation += "**Target:** Consider accumulating for long-term growth\n"
        recommendation += "**Risk Level:** Moderate (diversify position)\n"
        
    elif bullish_signals >= 3:
        recommendation += "👍 **BUY RECOMMENDATION**\n\n"
        recommendation += "**Why BUY:**\n"
        recommendation += "- Favorable technical setup for medium-term\n"
        recommendation += "- Reasonable risk-reward ratio\n"
        recommendation += "- Some positive momentum indicators\n"
        recommendation += "**Target:** Moderate position size\n"
        recommendation += "**Risk Level:** Medium (monitor closely)\n"
        
    elif bullish_signals >= 2:
        recommendation += "🤔 **HOLD / WAIT FOR BETTER ENTRY**\n\n"
        recommendation += "**Why HOLD:**\n"
        recommendation += "- Mixed technical signals\n"
        recommendation += "- Wait for clearer direction\n"
        recommendation += "- Consider averaging in on dips\n"
        recommendation += "**Action:** Maintain current position, watch for improvements\n"
        recommendation += "**Risk Level:** Medium-High\n"
        
    else:
        recommendation += "⏸️ **AVOID / CONSIDER SELLING**\n\n"
        recommendation += "**Why AVOID:**\n"
        recommendation += "- Multiple bearish indicators\n"
        recommendation += "- Weak price momentum\n"
        recommendation += "- Consider reducing exposure\n"
        recommendation += "**Action:** Wait for technical improvement or consider alternatives\n"
        recommendation += "**Risk Level:** High\n"
    
    # Additional considerations
    recommendation += "\n## 💡 Additional Considerations:\n\n"
    recommendation += "- **Market Conditions:** Monitor overall market trend\n"
    recommendation += "- **Sector Performance:** Consider sector-specific factors\n"
    recommendation += "- **Earnings:** Watch for upcoming earnings reports\n"
    recommendation += "- **Economic Data:** Monitor macroeconomic indicators\n"
    recommendation += "- **Diversification:** Never put all eggs in one basket\n"
    
    return recommendation

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period, interval='1d'):
    try:
        # Handle different symbol formats
        if not symbol.endswith('.NS') and symbol not in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']:
            symbol += '.NS'  # Assume Indian stock if not a common US stock

        # Adjust interval for real-time data
        if period in ['1d', '5d'] and interval == '1d':
            interval = '5m'  # Use 5-minute intervals for intraday data

        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)

        if hist.empty and symbol.endswith('.NS'):
            # Try without .NS suffix if it fails
            symbol = symbol.replace('.NS', '')
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)

        if hist.empty:
            st.info(f"Real-time data for {symbol} is currently unavailable. Using sample data for demonstration.")
            # Generate sample data for demonstration
            return generate_sample_data(symbol, period)

        return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        st.info("This could be due to temporary Yahoo Finance API issues. Using sample data for demonstration.")
        # Generate sample data for demonstration
        return generate_sample_data(symbol, period)

def generate_sample_data(symbol, period):
    """Generate sample stock data for demonstration when real data is unavailable"""
    try:
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
        
        st.info(f"Using sample data for {symbol} (real data unavailable)")
        return df
        
    except Exception as e:
        st.error(f"Error generating sample data: {e}")
        return None

# Technical indicators
def calculate_technical_indicators(df):
    df = df.copy()
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
