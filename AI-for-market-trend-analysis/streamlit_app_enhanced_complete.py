import streamlit as st
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
import requests

# Security functions
def generate_secure_hash(data, secret_key):
    """Generate HMAC SHA256 for data integrity"""
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
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
    sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
    signal_line = df['Signal_Line'].iloc[-1] if 'Signal_Line' in df.columns else 0

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
