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
    sma_20 = df['SMA_20'].iloc[-1]
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
    price_change_6m = (current_price / df['Close'].iloc[-180] - 1) * 100 if len(df) >= 180 else 0
    
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
    
    if current_price > sma_20:
        recommendation += "✅ **Price above 20-day SMA** - Short-term bullish\n"
        recommendation += f"   - Current: ${current_price:.2f} vs 20-SMA: ${sma_20:.2f}\n"
        recommendation += "   - **Why this matters:** Short-term momentum is positive\n"
    else:
        recommendation += "⚠️ **Price below 20-day SMA** - Short-term bearish\n"
        recommendation += f"   - Current: ${current_price:.2f} vs 20-SMA: ${sma_20:.2f}\n"
        recommendation += "   - **Why this matters:** Short-term momentum is negative\n"
    
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
        recommendation += "   - **Why this matters:** Positive momentum indicator suggesting upward trend\n"
    else:
        recommendation += "⚠️ **MACD below Signal Line** - Bearish momentum\n"
        recommendation += f"   - MACD: {macd:.4f}, Signal: {signal_line:.4f}\n"
        recommendation += "   - **Why this matters:** Negative momentum indicator suggesting downward pressure\n"
    
    # Volume analysis
    recommendation += "\n### 📊 Volume Analysis:\n\n"
    if volume_ratio > 1.5:
        recommendation += "🔥 **High volume activity** - Strong investor interest\n"
        recommendation += f"   - Recent volume: {volume_ratio:.1f}x average\n"
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
    
    if price_change_6m > 0:
        recommendation += f"📈 **6-month performance: +{price_change_6m:.1f}%** - Positive long-term momentum\n"
    else:
        recommendation += f"📉 **6-month performance: {price_change_6m:.1f}%** - Negative long-term momentum\n"
    
    # Final recommendation with detailed reasoning
    recommendation += "\n## 🎯 Investment Recommendation:\n\n"
    
    bullish_signals = sum([
        current_price > sma_50,
        sma_200 is not None and current_price > sma_200,
        rsi < 40,
        volume_ratio > 1.2,
        macd > signal_line,
        price_change_1m > 0,
        price_change_3m > 0
    ])
    
    if bullish_signals >= 6:
        recommendation += "🔥 **STRONG BUY RECOMMENDATION**\n\n"
        recommendation += "**Why BUY now:**\n"
        recommendation += "- Multiple bullish technical indicators align perfectly\n"
        recommendation += "- Strong price momentum across all timeframes\n"
        recommendation += "- High volume suggests institutional interest\n"
        recommendation += "- Oversold conditions provide excellent entry point\n"
        recommendation += "- All moving averages showing bullish alignment\n"
        recommendation += "**Target:** Consider accumulating for long-term growth\n"
        recommendation += "**Risk Level:** Moderate (diversify position)\n"
        recommendation += "**Time Horizon:** 6-12 months for optimal returns\n"
        
    elif bullish_signals >= 4:
        recommendation += "👍 **BUY RECOMMENDATION**\n\n"
        recommendation += "**Why BUY:**\n"
        recommendation += "- Favorable technical setup for medium-term growth\n"
        recommendation += "- Reasonable risk-reward ratio\n"
        recommendation += "- Multiple positive momentum indicators\n"
        recommendation += "- Good volume support for current price levels\n"
        recommendation += "**Target:** Moderate position size with stop-loss\n"
        recommendation += "**Risk Level:** Medium (monitor closely)\n"
