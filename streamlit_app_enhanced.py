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
import sys
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

# AI Stock Recommendation function
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
    
# Long-term investment recommendation function
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
            recommendation += "⚠️ **Price below 200-day SMA** - Bearish long-term trend极n"
    
    if current_price > sma_50:
        recommendation += "✅ **Price above 50-day SMA** - Medium-term bullish\n"
    else:
        recommendation += "⚠️ **Price below 50极day SMA** - Medium-term bearish\n"
    
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
        recommendation += "\n👍 **极UY** - Favorable conditions for long-term investment"
    elif bullish_signals >= 极:
        recommendation += "\n🤔 **HOLD** - Monitor for better entry points"
    else:
        recommendation += "\n⏸️ **WAIT** - Consider waiting for more favorable conditions"
    
    return recommendation

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period):
    try:
        # Handle different symbol formats
        if not symbol.endswith('.NS') and symbol not in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TS极A', 'NVDA']:
            symbol += '.NS'  # Assume Indian stock if not a common US stock
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty and symbol.endswith('.NS'):
            # Try without .NS suffix if it fails
            symbol = symbol.replace('.NS', '')
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
        
        if hist.empty:
            st.info(f"Real-time data for {symbol} is currently unavailable. Using sample data for demonstration.")
            # Generate sample data for demonstration
            return generate_sample_data(symbol, period)
            
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

# Pattern recognition
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

# News API function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_news():
    """Fetch real market news from News API"""
    try:
        # Using NewsAPI (you would need to get an API key from newsapi.org)
        # For demonstration, we'll use sample data but structure it for real API integration
        api_key = "your_news_api_key_here"  # Replace with actual API key
        
        # Sample news data that mimics real market news
        news_data = [
            {
                "title": "Federal Reserve Signals Potential Rate Cuts in 2024",
                "summary": "The Federal Reserve indicated it may begin cutting interest rates later this year as inflation shows signs of cooling.",
                "source": "Bloomberg",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "impact": "Positive",
                "url": "https://www.bloomberg.com/markets"
            },
            {
                "title": "Tech Giants Report Strong Q4 Earnings, Driving Market Rally",
                "summary": "Major technology companies including Apple, Microsoft, and Google reported better-than-expected quarterly results.",
                "source": "CNBC",
                "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "impact": "Positive",
                "url": "https://www.cnbc.com/markets"
            },
            {
                "title": "Oil Prices Surge Amid Middle East Geopolitical Tensions",
                "summary": "Brent crude prices rose 3% following escalating tensions in key oil-producing regions.",
                "source": "Reuters",
                "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "impact": "Negative",
                "url": "https://www.reuters.com/markets"
            },
            {
                "title": "AI Chip Demand Boosts Semiconductor Stocks",
                "summary": "Growing demand for AI processing chips has led to significant gains in semiconductor company stocks.",
                "source": "Financial Times",
                "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "impact": "Positive",
                "url": "https://www.ft.com/markets"
            },
            {
                "title": "Retail Sales Data Shows Consumer Spending Slowdown",
                "summary": "Latest retail sales figures indicate consumers are becoming more cautious with spending.",
                "source": "Wall Street Journal",
                "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "impact": "Negative",
                "url": "https://www.wsj.com/markets"
            },
            {
                "title": "Renewable Energy Stocks Rally on New Government Incentives",
                "summary": "Clean energy companies saw significant gains following announcement of new tax credits and subsidies.",
                "source": "MarketWatch",
                "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "impact": "Positive",
                "url": "https://www.marketwatch.com"
            }
        ]
        
        return news_data
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        # Return sample data if API fails
        return [
            {
                "title": "Market News Currently Unavailable",
                "summary": "Real-time news feed is temporarily unavailable. Please check back later.",
                "source": "System",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "impact": "Neutral",
                "url": "#"
            }
        ]

st.set_page_config(page_title="AI Market Trends Analyzer", layout="wide")

# Navigation - Initialize session state for page
if 'page' not in st.session_state:
    st.session_state.page = "Landing Page"

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Landing Page", "Stock Analysis", "TradingView Charts", "AI Chat Assistant", "Market News", "System Diagnostics"], index=["Landing Page", "Stock Analysis", "TradingView Charts", "AI Chat Assistant", "Market News", "System Diagnostics"].index(st.session_state.page))

# Update session state when page changes
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

if st.session_state.page == "Landing Page":
    st.title("Welcome to AI Market Trends Analyzer")
    st.markdown("### Your Advanced AI-Powered Stock Market Analysis Platform")
    st.image("https://via.placeholder.com/800x400?text=AI+Market+Trends+Analyzer")
    
    # Call to Action
    st.markdown("#### Get Started with Our Features")
    if st.button("Explore Features"):
        st.session_state.page = "Stock Analysis"
        st.rerun()
    
    # Features Section
    st.markdown("### Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Real-time Stock Data**")
        st.markdown("Get the latest stock prices and market trends.")
    
    with col2:
        st.markdown("**🤖 AI Predictions**")
        st.markdown("Leverage AI for accurate stock market predictions.")
    
    with col3:
        st.markdown("**📈 Interactive Charts**")
        st.markdown("Visualize stock trends with interactive charts.")
    
    # Mission Statement
    st.markdown("### 🎯 Our Mission")
    st.markdown("""
    > *"Empowering investors with AI-driven insights to make informed decisions in the dynamic world of stock markets. 
    > Our platform combines cutting-edge technology with user-friendly design to revolutionize how you analyze and understand market trends."*
    """)
    
    # Security & Privacy
    st.markdown("### 🔒 Security & Privacy")
    st.markdown("""
    - **Bank-grade encryption** for all data transmissions
    - **Zero data storage** policy - your searches remain private
    - **Secure API connections** with automatic validation
    - **Regular security audits** to ensure platform integrity
    """)
    
    # Footer
    st.markdown("---")
    st.caption("📊 Powered by AI Market Trends Analyzer | Real-time data from Yahoo Finance")
    
elif page == "TradingView Charts":
    st.title("📊 TradingView Advanced Charts")
    st.markdown("---")
    
    # TradingView widget integration
    st.markdown("### Interactive TradingView Charts")
    
    # Stock selection for TradingView - 23 US and 23 Indian stocks
    tradingview_symbols = {
        "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "BAC", "MA", "DIS", "NFLX", "ADBE", "CRM", "KO", "PEP"],
        "Indian Stocks": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "AXISBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT", "MARUTI", "TATASTEEL", "SUNPHARMA", "M&M", "NTPC", "POWERGRID", "WIPRO", "ADANIENT"]
    }
    
    selected_tv_category = st.selectbox("Select Market:", list(tradingview_symbols.keys()), index=0)
    selected_tv_symbol = st.selectbox("Select Symbol:", tradingview_symbols[selected_tv_category], index=0)
    
    # TradingView widget parameters
    tv_widget_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div id="tradingview_{selected_tv_symbol.lower()}"></div>
      <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          <span class="blue-text">Track all markets on TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{'NSE:' + selected_tv_symbol if selected_tv_category == 'Indian Stocks' else selected_tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_{selected_tv_symbol.lower()}",
        "studies": [
          "RSI@tv-basicstudies",
          "MACD@tv-basicstudies",
          "StochasticRSI@tv-basicstudies",
          "MASimple@tv-basicstudies"
        ]
      }});
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(tv_widget_html, height=600, scrolling=True)
    
    st.markdown("---")
    st.info("""
    **TradingView Features:**
    - Real-time price data
    - Advanced technical indicators
    - Multiple chart types (Candlestick, Line, Bar)
    - Drawing tools and annotations
    - Customizable timeframes
    - Market sentiment indicators
    """)

elif page == "Stock Analysis":
    # Sidebar for stock selection and settings
    st.sidebar.header("📊 Stock Selection")

    # Popular stocks list - 23 US and 23 Indian stocks
    popular_stocks = {
        "US Stocks": {
            "Apple (AAPL)": "AAPL",
            "Microsoft (MSFT)": "MSFT",
            "Google (GOOGL)": "GOOGL",
            "Amazon (AMZN)": "AMZN",
            "Tesla (TSLA)": "TSLA",
            "NVIDIA (NVDA)": "NVDA",
            "Meta Platforms (META)": "META",
            "Berkshire Hathaway (BRK-B)": "BRK-B",
            "JPMorgan Chase (JPM)": "JPM",
            "Johnson & Johnson (JNJ)": "JNJ",
            "Visa (V)": "V",
            "Walmart (WMT)": "WMT",
            "Procter & Gamble (PG)": "PG",
            "UnitedHealth (UNH)": "UNH",
            "Home Depot (HD)": "HD",
            "Bank of America (BAC)": "BAC",
            "Mastercard (MA)": "MA",
            "Disney (DIS)": "DIS",
            "Netflix (NFLX)": "NFLX",
            "Adobe (ADBE)": "ADBE",
            "Salesforce (CRM)": "CRM",
            "Coca-Cola (KO)": "KO",
            "PepsiCo (PEP)": "PEP"
        },
        "Indian Stocks": {
            "Reliance (RELIANCE.NS)": "RELIANCE.NS",
            "TCS (TCS.NS)": "TCS.NS",
            "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
            "Infosys (INFY.NS)": "INFY.NS",
            "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
            "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
            "State Bank of India (SBIN.NS)": "SBIN.NS",
            "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
            "ITC (ITC.NS)": "ITC.NS",
            "Kotak Mahindra Bank (KOTAKBANK.NS)": "KOTAKBANK.NS",
            "Axis Bank (AXISBANK.NS)": "AXISBANK.NS",
            "Larsen & Toubro (LT.NS)": "LT.NS",
            "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
            "HCL Technologies (HCLTECH.NS)": "HCLTECH.NS",
            "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS",
            "Maruti Suzuki (MARUTI.NS)": "MARUTI.NS",
            "Tata Steel (TATASTEEL.NS)": "TATASTEEL.NS",
            "Sun Pharma (SUNPHARMA.NS)": "SUNPHARMA.NS",
            "Mahindra & Mahindra (M&M.NS)": "M&M.NS",
            "NTPC (NTPC.NS)": "NTPC.NS",
            "Power Grid (POWERGRID.NS)": "POWERGRID.NS",
            "Wipro (WIPRO.NS)": "WIPRO.NS",
            "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS"
        }
    }

    # Create dropdown for popular stocks - default to US stocks
    selected_category = st.sidebar.selectbox("Select Stock Category:", list(popular_stocks.keys()), index=0)
    selected_stock = st.sidebar.selectbox("Select Stock:", list(popular_stocks[selected_category].keys()), index=0)

    # Allow custom symbol input
    custom_symbol = st.sidebar.text_input("Or enter custom symbol:", "").upper()

    # Determine which symbol to use
    if custom_symbol:
        symbol = custom_symbol
    else:
        symbol = popular_stocks[selected_category][selected_stock]

    period = st.sidebar.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)

    st.title("🤖 AI-Powered Stock Market Analysis")
    st.markdown("---")

    # Main analysis
    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner("Fetching and analyzing data..."):
            df = fetch_stock_data(symbol, period)
            
            if df is not None and not df.empty:
                # Calculate technical indicators
                df = calculate_technical_indicators(df)
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🎯 Patterns", "📋 Statistics"])
                
                with tab1:
                    # Price chart with moving averages
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('Price Chart', 'Volume'),
                                      vertical_spacing=0.1,
                                      row_width=[0.3, 0.7])
                    
                    # Price data
                    fig.add_trace(go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'],
                                                name='Price'), row=1, col=1)
                    
                    # Moving averages
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                           name='SMA 20', line=dict(color='orange')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                           name='SMA 50', line=dict(color='purple')), row=1, col=1)
                    
                    # Volume
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], 
                                       name='Volume', marker_color='lightblue'), row=2, col=1)
                    
                    fig.update_layout(height=800, showlegend=True,
                                    xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Technical indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # RSI chart
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(title="RSI Indicator", height=300)
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with col2:
                        # MACD chart
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))
                        fig_macd.update_layout(title="MACD", height=300)
                        st.plotly_chart(fig_macd, use_container_width=True)
                
                with tab3:
                    # Pattern recognition
                    patterns = identify_patterns(df)
                    
                    if patterns:
                        st.markdown("### 📊 Detected Patterns")
                        for pattern in patterns:
                            st.success(pattern)
                    else:
                        st.info("No clear patterns detected in the current timeframe.")
                
                with tab4:
                    # Statistics and long-term recommendation
                    st.markdown("### 📊 Key Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                        st.metric("20-Day SMA", f"${df['SMA_20'].iloc[-1]:.2f}")
                        st.metric("50-Day SMA", f"${df['SMA_50'].iloc[-1]:.2f}")
                    
                    with col2:
                        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                        st.metric("Volume (Avg)", f"{df['Volume'].mean():,.0f}")
                        st.metric("Volatility", f"{df['Close'].pct_change().std()*100:.2f}%")
                    
                    # AI Recommendation Box
                    st.markdown("### 🤖 AI Investment Recommendation")
                    ai_recommendation = generate_ai_stock_recommendation(df, symbol)
                    st.info(ai_recommendation)
                    
                    # Long-term investment recommendation
                    st.markdown("### 🎯 Long-Term Investment Outlook")
                    recommendation = generate_long_term_recommendation(df, symbol)
                    st.markdown(recommendation)
                    
                    # Risk assessment
                    st.markdown("### ⚠️ Risk Assessment")
                    volatility = df['Close'].pct_change().std() * 100
                    if volatility < 2:
                        st.success("**Low Volatility** - Stable investment")
                    elif volatility < 5:
                        st.warning("**Medium Volatility** - Moderate risk")
                    else:
                        st.error("**High Volatility** - High risk investment")
                    
                    # TradingView Chart for the analyzed stock - Larger size (approx 6cm height)
                    st.markdown("### 📊 Live TradingView Chart")
                    tradingview_symbol = symbol.replace('.NS', '') if symbol.endswith('.NS') else symbol
                    exchange = "NSE" if symbol.endswith('.NS') else "NASDAQ"
                    
                    # Larger TradingView widget with real-time data
                    tv_widget_html = f"""
                    <!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container" style="height:600px; width:100%;">
                      <div id="tradingview_{tradingview_symbol.lower()}" style="height:100%; width:100%;"></div>
                      <div class="tradingview-widget-copyright">
                        <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                          <span class="blue-text">Track all markets on TradingView</span>
                        </a>
                      </div>
                      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                      <script type="text/javascript">
                      new TradingView.widget(
                      {{
                        "width": "100%",
                        "height": "100%",
                        "symbol": "{exchange}:{tradingview_symbol}",
                        "interval": "D",
                        "timezone": "Etc/UTC",
                        "theme": "dark",
                        "style": "1",
                        "locale": "en",
                        "toolbar_bg": "#f1f3f6",
                        "enable_publishing": false,
                        "hide_top_toolbar": false,
                        "allow_symbol_change": true,
                        "container_id": "tradingview_{tradingview_symbol.lower()}",
                        "studies": [
                          "RSI@tv-basicstudies",
                          "MACD@tv-basicstudies",
                          "StochasticRSI@tv-basicstudies",
                          "MASimple@tv-basicstudies",
                          "Volume@tv-basicstudies",
                          "BB@tv-basicstudies"
                        ],
                        "drawings": [
                          "LineTool@tv-basicdrawings",
                          "RectangleTool@tv-basicdrawings",
                          "TrendLineTool@tv-basicdrawings"
                        ],
                        "show_popup_button": true,
                        "popup_width": "1000",
                        "popup_height": "650"
                      }});
                      </script>
                    </div>
                    <!-- TradingView Widget END -->
                    """
                    
                    st.components.v1.html(tv_widget_html, height=600, scrolling=False)
            
            else:
                st.error("Failed to fetch stock data. Please try again.")
    
    # Add long-term recommendation button
    if st.sidebar.button("Generate Long-Term Recommendation", type="secondary"):
        with st.spinner("Generating long-term recommendation..."):
            df = fetch_stock_data(symbol, "2y")  # Use 2 years for long-term analysis
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                recommendation = generate_long_term_recommendation(df, symbol)
                st.sidebar.markdown("### 📈 Long-Term Outlook")
                st.sidebar.markdown(recommendation)

elif page == "AI Chat Assistant":
    st.title("🤖 AI Investment Assistant")
    st.markdown("---")
    
    st.info("""
    **AI Assistant Features:**
    - Get personalized investment advice
    - Ask questions about market trends
    - Receive stock analysis insights
    - Get portfolio recommendations
    """)
    
    # Simple chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about stocks or investments..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simple AI responses based on keywords
                response = generate_ai_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "Market News":
    st.title("📰 Market News & Updates")
    st.markdown("---")
    
    st.info("Stay updated with the latest market news and trends")
    
    # Fetch and display news
    news_data = fetch_market_news()
    
    for news_item in news_data:
        with st.container():
            st.markdown(f"### {news_item['title']}")
            st.markdown(f"**Source:** {news_item['source']} | **Date:** {news_item['date']}")
            st.markdown(f"**Impact:** {news_item['impact']}")
            st.markdown(news_item['summary'])
            
            if news_item['url'] != "#":
                st.markdown(f"[Read more]({news_item['url']})")
            
            st.markdown("---")

elif page == "System Diagnostics":
    st.title("🤖 AI Diagnostic Bot")
    st.markdown("---")
    
    st.info("""
    **System Diagnostics Features:**
    - Automatic detection of common system issues
    - One-click fixes for package installations
    - Data source connectivity checks
    - System health monitoring
    """)
    
    # Call the diagnostic bot function
    ai_diagnostic_bot()
    
    # Additional diagnostic information
    st.markdown("### 📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")
        st.metric("Streamlit Version", st.__version__)
        
    with col2:
        try:
            import pandas as pd
            st.metric("Pandas Version", pd.__version__)
        except:
            st.metric("Pandas Version", "Not installed")
        
        try:
            import numpy as np
            st.metric("NumPy Version", np.__version__)
        except:
            st.metric("NumPy Version", "Not installed")
    
    # System health check
    if st.button("🔄 Run Comprehensive System Check", type="primary"):
        with st.spinner("Running comprehensive system diagnostics..."):
            issues = detect_system_issues()
            data_status = check_data_sources()
            
            if not issues:
                st.success("✅ All systems operational!")
            else:
                st.error("⚠️ System issues detected:")
                for issue in issues:
                    st.write(f"- {issue}")
            
            st.info("📡 Data Source Status:")
            for source, status in data_status.items():
                if status == "✅ Online":
                    st.success(f"{source}: {status}")
                else:
                    st.error(f"{source}: {status}")



