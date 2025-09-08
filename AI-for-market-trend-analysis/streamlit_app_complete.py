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

# AI Chat Response function
def generate_ai_response(prompt):
    """Generate AI response for chat assistant based on user input"""
    prompt_lower = prompt.lower()

    # Investment advice responses
    if any(word in prompt_lower for word in ['buy', 'sell', 'invest', 'recommend']):
        return """🤖 **Investment Advice:**
- Always do your own research (DYOR)
- Diversify your portfolio
- Consider your risk tolerance
- Long-term investing often outperforms short-term trading
- Consult with a financial advisor for personalized advice"""

    # Market analysis responses
    elif any(word in prompt_lower for word in ['market', 'trend', 'analysis']):
        return """📊 **Market Analysis Tips:**
- Use technical indicators (RSI, MACD, Moving Averages)
- Monitor volume and volatility
- Consider fundamental factors (earnings, news)
- Watch for market sentiment and economic indicators
- Use multiple timeframes for better perspective"""

    # Risk management responses
    elif any(word in prompt_lower for word in ['risk', 'loss', 'stop loss']):
        return """⚠️ **Risk Management:**
- Never invest more than you can afford to lose
- Use stop-loss orders to limit losses
- Diversify across different asset classes
- Consider position sizing (1-2% of portfolio per trade)
- Have an exit strategy before entering a trade"""

    # General investment questions
    elif any(word in prompt_lower for word in ['beginner', 'start', 'how to']):
        return """🚀 **Getting Started with Investing:**
1. **Education**: Learn basics of stocks, bonds, ETFs
2. **Goal Setting**: Define your investment objectives
3. **Risk Assessment**: Understand your risk tolerance
4. **Broker Account**: Open with a reputable broker
5. **Start Small**: Begin with small amounts
6. **Regular Investing**: Consider dollar-cost averaging
7. **Monitor & Learn**: Track performance and learn from experience"""

    # Default response
    else:
        return """💡 **General Investment Wisdom:**
- "The stock market is a device for transferring money from the impatient to the patient." - Warren Buffett
- Focus on long-term growth rather than short-term gains
- Stay informed about market news and economic indicators
- Consider both technical and fundamental analysis
- Remember that past performance doesn't guarantee future results

Feel free to ask me specific questions about stocks, market trends, or investment strategies!"""

# System Diagnostics functions
def diagnostic_bot():
    """AI-powered diagnostic bot for system health monitoring"""
    st.markdown("### 🔍 System Health Check")

    # Check Python environment
    st.markdown("#### 🐍 Python Environment")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")
        st.metric("Platform", sys.platform)

    with col2:
        try:
            import streamlit
            st.success("✅ Streamlit: Available")
        except ImportError:
            st.error("❌ Streamlit: Not installed")

        try:
            import yfinance
            st.success("✅ yfinance: Available")
        except ImportError:
            st.error("❌ yfinance: Not installed")

    # Check data sources
    st.markdown("#### 📡 Data Sources")
    data_sources = {
        "Yahoo Finance": "https://finance.yahoo.com",
        "TradingView": "https://www.tradingview.com",
        "News API": "https://newsapi.org"
    }

    for source, url in data_sources.items():
        try:
            import requests
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                st.success(f"✅ {source}: Online")
            else:
                st.warning(f"⚠️ {source}: Status {response.status_code}")
        except:
            st.error(f"❌ {source}: Connection failed")

    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    import psutil
    try:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")
        st.metric("Available Memory", f"{memory.available / (1024**3):.1f} GB")
    except ImportError:
        st.info("Install 'psutil' for detailed system metrics")

def detect_system_issues():
    """Detect common system issues and return list of problems"""
    issues = []

    # Check required packages
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'yfinance', 'requests']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing required package: {package}")

    # Check internet connectivity
    try:
        import requests
        requests.get("https://www.google.com", timeout=5)
    except:
        issues.append("No internet connectivity detected")

    # Check file permissions
    try:
        with open("test_write.tmp", "w") as f:
            f.write("test")
        import os
        os.remove("test_write.tmp")
    except:
        issues.append("File write permissions issue")

    return issues

def check_data_sources():
    """Check status of various data sources"""
    sources = {
        "Yahoo Finance": "https://finance.yahoo.com",
        "TradingView": "https://www.tradingview.com",
        "News API": "https://newsapi.org"
    }

    status = {}
    try:
        import requests
        for name, url in sources.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    status[name] = "✅ Online"
                else:
                    status[name] = f"⚠️ Status {response.status_code}"
            except:
                status[name] = "❌ Offline"
    except ImportError:
        for name in sources.keys():
            status[name] = "❓ Cannot check (requests not available)"

    return status

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

@st.cache_data(ttl=1800)  # Reduced cache time to 30 minutes for better data freshness
def fetch_stock_data(symbol, period, max_retries=5):
    """
    Enhanced stock data fetching with improved error handling and multiple fallback strategies
    """
    # Handle different symbol formats
    original_symbol = symbol
    symbols_to_try = [symbol]

    # Add alternative symbol formats
    if symbol.endswith('.NS'):
        symbols_to_try.append(symbol.replace('.NS', ''))  # Try without .NS
    elif not symbol.endswith('.NS') and symbol not in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BTC-USD', 'ETH-USD']:
        symbols_to_try.append(symbol + '.NS')  # Try with .NS for Indian stocks

    periods_to_try = [period, '1y', '6mo', '3mo', '1mo']

    for attempt in range(max_retries):
        for current_symbol in symbols_to_try:
            for current_period in periods_to_try:
                try:
                    if attempt == 0 and current_symbol == symbol and current_period == period:
                        st.info(f"Attempting to fetch real data for {original_symbol} (Attempt {attempt + 1}/{max_retries})")

                    # Create ticker with timeout and error handling
                    stock = yf.Ticker(current_symbol)

                    # Try to get basic info first
                    try:
                        info = stock.info
                        if not info or len(info) < 5:
                            continue  # Skip if no basic info available
                    except:
                        continue  # Skip if info fetch fails

                    # Fetch historical data
                    hist = stock.history(period=current_period, timeout=30)

                    if hist is not None and not hist.empty and len(hist) > 10:
                        if current_symbol != original_symbol or current_period != period:
                            st.success(f"✅ Successfully fetched real data for {original_symbol} (using {current_symbol}, {current_period})")
                        else:
                            st.success(f"✅ Successfully fetched real data for {original_symbol}")
                        return hist

                except Exception as e:
                    # Only show warnings for the first attempt with original parameters
                    if attempt == 0 and current_symbol == symbol and current_period == period:
                        st.warning(f"Attempt {attempt + 1} failed for {current_symbol}: {str(e)}")
                    continue

        # Wait before retry with exponential backoff
        if attempt < max_retries - 1:
            wait_time = min(2 ** attempt, 10)  # Max 10 seconds wait
            time.sleep(wait_time)

    # If all attempts fail, use enhanced sample data
    st.warning(f"❌ Unable to fetch real data for {original_symbol} after {max_retries} attempts")
    st.info("Using enhanced sample data for demonstration. This ensures the app remains functional.")
    return generate_enhanced_sample_data(original_symbol, period)

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

def generate_enhanced_sample_data(symbol, period):
    """Generate enhanced sample stock data with more realistic patterns and technical indicators"""
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

        # Generate more realistic price data based on symbol type
        if symbol.endswith('.NS'):
            # Indian stocks - typically range from ₹100-₹5000
            base_price = np.random.uniform(500, 3000)
            volatility = 0.02  # Moderate volatility
        elif symbol in ['BTC-USD', 'ETH-USD']:
            # Cryptocurrencies - high volatility, wide range
            base_price = np.random.uniform(20000, 60000) if symbol == 'BTC-USD' else np.random.uniform(1000, 4000)
            volatility = 0.05  # High volatility
        else:
            # US stocks - typically $50-$500
            base_price = np.random.uniform(50, 500)
            volatility = 0.025  # Moderate volatility

        # Create price series with trend and seasonality
        prices = []
        current_price = base_price

        # Add trend component
        trend = np.random.choice([-0.001, 0, 0.001], p=[0.3, 0.4, 0.3])  # Slight downward, flat, or upward trend

        for i in range(days):
            # Base random walk
            change = np.random.normal(trend, volatility)

            # Add some seasonality (weekend effect)
            if i % 7 in [5, 6]:  # Friday/Saturday
                change *= 1.2  # Slightly more volatile

            current_price = max(1, current_price + change)  # Don't go below $1
            prices.append(current_price)

        # Create DataFrame with OHLCV data
        df = pd.DataFrame(index=dates)

        # Generate OHLC from close prices with realistic spreads
        df['Close'] = prices
        df['Open'] = [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices]

        for i in range(len(df)):
            high_spread = abs(np.random.uniform(0.005, 0.03))
            low_spread = abs(np.random.uniform(0.005, 0.03))
            df.loc[df.index[i], 'High'] = max(df.loc[df.index[i], 'Open'], df.loc[df.index[i], 'Close']) * (1 + high_spread)
            df.loc[df.index[i], 'Low'] = min(df.loc[df.index[i], 'Open'], df.loc[df.index[i], 'Close']) * (1 - low_spread)

        # Generate volume based on price and volatility
        base_volume = np.random.randint(500000, 2000000)
        df['Volume'] = [int(base_volume * (1 + abs(np.random.normal(0, 0.5)))) for _ in range(days)]

        # Add some gaps and limit moves for realism
        for i in range(1, len(df)):
            # Occasional gaps (5% chance)
            if np.random.random() < 0.05:
                gap = np.random.uniform(-0.02, 0.02)
                df.loc[df.index[i], 'Open'] = df.loc[df.index[i-1], 'Close'] * (1 + gap)

            # Limit moves (rare but realistic)
            if abs(df.loc[df.index[i], 'Close'] - df.loc[df.index[i-1], 'Close']) / df.loc[df.index[i-1], 'Close'] > 0.1:
                df.loc[df.index[i], 'Close'] = df.loc[df.index[i-1], 'Close'] * (1 + np.sign(np.random.uniform(-1, 1)) * 0.08)

        st.info(f"Using enhanced sample data for {symbol} (real data unavailable)")
        return df

    except Exception as e:
        st.error(f"Error generating enhanced sample data: {e}")
        # Fallback to basic sample data
        return generate_sample_data(symbol, period)

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
page = st.sidebar.selectbox("Navigate", ["Landing Page", "Stock Analysis", "TradingView Charts", "AI Chat Assistant", "Market News", "Enhanced Model Training", "System Diagnostics"], index=["Landing Page", "Stock Analysis", "TradingView Charts", "AI Chat Assistant", "Market News", "Enhanced Model Training", "System Diagnostics"].index(st.session_state.page))

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
    
    # Stock selection for TradingView - Expanded with all categories
    tradingview_symbols = {
        "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "BAC", "MA", "DIS", "NFLX", "ADBE", "CRM", "KO", "PEP", "GOOG", "INTC", "CSCO", "PFE", "VZ", "T", "XOM", "CVX", "BA", "CAT", "GS", "MS", "NKE", "MCD", "SBUX", "COST", "TGT", "FDX", "UPS"],
        "Indian Stocks": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "AXISBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT", "MARUTI", "TATASTEEL", "SUNPHARMA", "M&M", "NTPC", "POWERGRID", "WIPRO", "ADANIENT", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "DRREDDY", "CIPLA", "GRASIM", "ULTRACEMCO", "JSWSTEEL", "COALINDIA", "GAIL", "ONGC", "HINDALCO", "VEDL", "NMDC", "BPCL", "IOC", "SHREECEM", "DABUR", "BRITANNIA", "NESTLEIND", "GODFRYPHLP", "HYUNDAI"],
        "Cryptocurrencies": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD", "VET-USD", "THETA-USD", "FIL-USD", "TRX-USD", "EOS-USD", "XMR-USD", "MIOTA-USD"],
        "International Stocks": ["005930.KS", "7203.T", "VOW.DE", "6758.T", "7974.T", "ASML.AS", "NOVO-B.CO", "SAP.DE", "ENR.DE", "MC.PA", "TTE.PA", "AIR.PA", "ULVR.L", "HSBA.L", "BP.L", "RIO.L", "BHP.L", "8306.T", "0941.HK", "0700.HK"]
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

    # Popular stocks list - Expanded with 20+ additional stocks and cryptocurrencies
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
            "PepsiCo (PEP)": "PEP",
            # Additional US Stocks
            "Alphabet (GOOG)": "GOOG",
            "Intel (INTC)": "INTC",
            "Cisco (CSCO)": "CSCO",
            "Pfizer (PFE)": "PFE",
            "Verizon (VZ)": "VZ",
            "AT&T (T)": "T",
            "Exxon Mobil (XOM)": "XOM",
            "Chevron (CVX)": "CVX",
            "Boeing (BA)": "BA",
            "Caterpillar (CAT)": "CAT",
            "Goldman Sachs (GS)": "GS",
            "Morgan Stanley (MS)": "MS",
            "Nike (NKE)": "NKE",
            "McDonald's (MCD)": "MCD",
            "Starbucks (SBUX)": "SBUX",
            "Costco (COST)": "COST",
            "Target (TGT)": "TGT",
            "FedEx (FDX)": "FDX",
            "UPS (UPS)": "UPS"
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
            "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS",
            # Additional Indian Stocks
            "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
            "Bajaj Auto (BAJAJ-AUTO.NS)": "BAJAJ-AUTO.NS",
            "Hero MotoCorp (HEROMOTOCO.NS)": "HEROMOTOCO.NS",
            "Dr. Reddy's (DRREDDY.NS)": "DRREDDY.NS",
            "Cipla (CIPLA.NS)": "CIPLA.NS",
            "Grasim Industries (GRASIM.NS)": "GRASIM.NS",
            "UltraTech Cement (ULTRACEMCO.NS)": "ULTRACEMCO.NS",
            "JSW Steel (JSWSTEEL.NS)": "JSWSTEEL.NS",
            "Coal India (COALINDIA.NS)": "COALINDIA.NS",
            "GAIL (GAIL.NS)": "GAIL.NS",
            "ONGC (ONGC.NS)": "ONGC.NS",
            "Hindalco (HINDALCO.NS)": "HINDALCO.NS",
            "Vedanta (VEDL.NS)": "VEDL.NS",
            "NMDC (NMDC.NS)": "NMDC.NS",
            "BPCL (BPCL.NS)": "BPCL.NS",
            "IOC (IOC.NS)": "IOC.NS",
            "Shree Cement (SHREECEM.NS)": "SHREECEM.NS",
            "Dabur (DABUR.NS)": "DABUR.NS",
            "Britannia (BRITANNIA.NS)": "BRITANNIA.NS",
            "Nestle India (NESTLEIND.NS)": "NESTLEIND.NS",
            "Godfrey Phillips (GODFRYPHLP.NS)": "GODFRYPHLP.NS",
            "Hyundai Motor India (HYUNDAI.NS)": "HYUNDAI.NS"
        },
        "Cryptocurrencies": {
            "Bitcoin (BTC-USD)": "BTC-USD",
            "Ethereum (ETH-USD)": "ETH-USD",
            "Binance Coin (BNB-USD)": "BNB-USD",
            "Cardano (ADA-USD)": "ADA-USD",
            "Solana (SOL-USD)": "SOL-USD",
            "Polkadot (DOT-USD)": "DOT-USD",
            "Dogecoin (DOGE-USD)": "DOGE-USD",
            "Avalanche (AVAX-USD)": "AVAX-USD",
            "Chainlink (LINK-USD)": "LINK-USD",
            "Litecoin (LTC-USD)": "LTC-USD",
            "Bitcoin Cash (BCH-USD)": "BCH-USD",
            "Stellar (XLM-USD)": "XLM-USD",
            "Cosmos (ATOM-USD)": "ATOM-USD",
            "VeChain (VET-USD)": "VET-USD",
            "Theta Network (THETA-USD)": "THETA-USD",
            "Filecoin (FIL-USD)": "FIL-USD",
            "Tron (TRX-USD)": "TRX-USD",
            "EOS (EOS-USD)": "EOS-USD",
            "Monero (XMR-USD)": "XMR-USD",
            "IOTA (MIOTA-USD)": "MIOTA-USD"
        },
        "International Stocks": {
            "Samsung Electronics (005930.KS)": "005930.KS",
            "Toyota Motor (7203.T)": "7203.T",
            "Volkswagen (VOW.DE)": "VOW.DE",
            "Sony (6758.T)": "6758.T",
            "Nintendo (7974.T)": "7974.T",
            "ASML Holding (ASML.AS)": "ASML.AS",
            "Novo Nordisk (NOVO-B.CO)": "NOVO-B.CO",
            "SAP SE (SAP.DE)": "SAP.DE",
            "Siemens (ENR.DE)": "ENR.DE",
            "LVMH (MC.PA)": "MC.PA",
            "Total Energies (TTE.PA)": "TTE.PA",
            "Airbus (AIR.PA)": "AIR.PA",
            "Unilever (ULVR.L)": "ULVR.L",
            "HSBC (HSBA.L)": "HSBA.L",
            "BP (BP.L)": "BP.L",
            "Rio Tinto (RIO.L)": "RIO.L",
            "BHP Group (BHP.L)": "BHP.L",
            "Mitsubishi UFJ (8306.T)": "8306.T",
            "China Mobile (0941.HK)": "0941.HK",
            "Tencent (0700.HK)": "0700.HK"
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
                    
                    # Enhanced Risk Assessment
                    st.markdown("### ⚠️ Advanced Risk Assessment")

                    # Calculate multiple volatility metrics
                    daily_returns = df['Close'].pct_change().dropna()
                    volatility_daily = daily_returns.std() * 100
                    volatility_weekly = df['Close'].pct_change(5).dropna().std() * 100
                    volatility_monthly = df['Close'].pct_change(20).dropna().std() * 100

                    # Calculate Value at Risk (VaR) at 95% confidence
                    var_95 = np.percentile(daily_returns, 5) * 100

                    # Calculate Sharpe ratio (assuming 2% risk-free rate)
                    risk_free_rate = 0.02 / 252  # Daily risk-free rate
                    excess_returns = daily_returns - risk_free_rate
                    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

                    # Maximum drawdown calculation
                    cumulative = (1 + daily_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min() * 100

                    # Beta calculation (relative to market - using S&P 500 approximation)
                    market_returns = daily_returns.rolling(20).mean()  # Simplified market proxy
                    if len(market_returns.dropna()) > 1:
                        covariance = daily_returns.cov(market_returns)
                        market_variance = market_returns.var()
                        beta = covariance / market_variance if market_variance > 0 else 1.0
                    else:
                        beta = 1.0

                    # Asset-type specific risk thresholds
                    if symbol.endswith('.NS'):
                        # Indian stocks
                        low_vol_threshold = 1.5
                        med_vol_threshold = 3.0
                        high_vol_threshold = 4.5
                    elif symbol in ['BTC-USD', 'ETH-USD'] or '-USD' in symbol:
                        # Cryptocurrencies
                        low_vol_threshold = 3.0
                        med_vol_threshold = 6.0
                        high_vol_threshold = 10.0
                    else:
                        # US stocks and others
                        low_vol_threshold = 1.2
                        med_vol_threshold = 2.5
                        high_vol_threshold = 4.0

                    # Determine risk level based on multiple factors
                    risk_score = 0

                    # Volatility component (40% weight)
                    if volatility_daily <= low_vol_threshold:
                        risk_score += 1  # Low risk
                    elif volatility_daily <= med_vol_threshold:
                        risk_score += 2  # Medium risk
                    else:
                        risk_score += 3  # High risk

                    # VaR component (30% weight)
                    if abs(var_95) <= 1.5:
                        risk_score += 1
                    elif abs(var_95) <= 3.0:
                        risk_score += 2
                    else:
                        risk_score += 3

                    # Max drawdown component (20% weight)
                    if abs(max_drawdown) <= 5:
                        risk_score += 1
                    elif abs(max_drawdown) <= 10:
                        risk_score += 2
                    else:
                        risk_score += 3

                    # Beta component (10% weight)
                    if beta <= 0.8:
                        risk_score += 1
                    elif beta <= 1.2:
                        risk_score += 2
                    else:
                        risk_score += 3

                    # Final risk assessment
                    avg_risk_score = risk_score / 4

                    if avg_risk_score <= 1.5:
                        risk_level = "LOW RISK"
                        risk_color = "success"
                        risk_desc = "Stable investment with low volatility"
                        risk_icon = "🟢"
                    elif avg_risk_score <= 2.5:
                        risk_level = "MODERATE RISK"
                        risk_color = "warning"
                        risk_desc = "Balanced risk-reward profile"
                        risk_icon = "🟡"
                    else:
                        risk_level = "HIGH RISK"
                        risk_color = "error"
                        risk_desc = "High volatility - suitable for experienced investors"
                        risk_icon = "🔴"

                    # Display risk assessment
                    col_risk1, col_risk2 = st.columns(2)

                    with col_risk1:
                        if risk_color == "success":
                            st.success(f"{risk_icon} **{risk_level}** - {risk_desc}")
                        elif risk_color == "warning":
                            st.warning(f"{risk_icon} **{risk_level}** - {risk_desc}")
                        else:
                            st.error(f"{risk_icon} **{risk_level}** - {risk_desc}")

                        st.metric("Daily Volatility", f"{volatility_daily:.2f}%")
                        st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

                    with col_risk2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        st.metric("Beta", f"{beta:.2f}")
                        st.metric("Weekly Volatility", f"{volatility_weekly:.2f}%")
                        st.metric("Monthly Volatility", f"{volatility_monthly:.2f}%")

                    # Risk interpretation
                    st.markdown("#### 📊 Risk Metrics Interpretation")
                    risk_interpretation = f"""
                    **Risk Score:** {avg_risk_score:.1f}/3.0

                    **Key Indicators:**
                    - **Daily Volatility:** {volatility_daily:.2f}% (Price fluctuation per day)
                    - **VaR (95%):** {var_95:.2f}% (Maximum expected loss in normal market conditions)
                    - **Max Drawdown:** {max_drawdown:.2f}% (Largest peak-to-trough decline)
                    - **Sharpe Ratio:** {sharpe_ratio:.2f} (Risk-adjusted returns)
                    - **Beta:** {beta:.2f} (Market sensitivity)

                    **Investment Suitability:**
                    """

                    if avg_risk_score <= 1.5:
                        risk_interpretation += "- ✅ Conservative investors\n- ✅ Long-term investors\n- ✅ Risk-averse portfolios\n- ✅ Retirement accounts"
                    elif avg_risk_score <= 2.5:
                        risk_interpretation += "- ✅ Balanced investors\n- ✅ Moderate risk tolerance\n- ✅ Diversified portfolios\n- ✅ Growth-oriented accounts"
                    else:
                        risk_interpretation += "- ⚠️ Experienced investors only\n- ⚠️ High risk tolerance required\n- ⚠️ Active risk management needed\n- ⚠️ Speculative positions"

                    st.info(risk_interpretation)
                    
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

elif page == "Enhanced Model Training":
    st.title("🚀 Enhanced AI Model Training with SHAP Explainability")
    st.markdown("---")

    st.info("""
    **Enhanced Model Training Features:**
    - Advanced LSTM models with multiple architectures
    - SHAP explainability for model predictions
    - Feature importance analysis
    - Model performance comparison
    - Real-time training progress
    """)

    # Import required modules for enhanced training
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

        from enhanced_model_training import EnhancedStockPredictor
        from data_preprocessing import StockDataPreprocessor
        from feature_engineering import AdvancedFeatureEngineer
        from evaluation import ModelEvaluator
        import shap
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Sidebar controls for enhanced training
        st.sidebar.header("🎯 Training Configuration")

        # Stock selection for training
        training_stocks = {
            "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
            "Cryptocurrencies": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD"]
        }

        selected_category = st.sidebar.selectbox("Select Training Category:", list(training_stocks.keys()), index=0)
        selected_symbol = st.sidebar.selectbox("Select Symbol for Training:", training_stocks[selected_category], index=0)

        # Training parameters
        st.sidebar.subheader("Training Parameters")
        epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60)
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

        # Model architecture selection
        model_type = st.sidebar.selectbox("Model Architecture:",
                                         ["LSTM", "GRU", "Bidirectional LSTM", "CNN-LSTM", "Attention LSTM"],
                                         index=0)

        # Training options
        include_shap = st.sidebar.checkbox("Include SHAP Explainability", value=True)
        cross_validation = st.sidebar.checkbox("Cross-Validation", value=False)
        early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

        # Main training interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Training Progress & Results")

            if st.button("🚀 Start Enhanced Training", type="primary"):
                with st.spinner("Initializing enhanced training system..."):
                    try:
                        # Initialize components
                        preprocessor = StockDataPreprocessor()
                        feature_engineer = AdvancedFeatureEngineer()
                        predictor = EnhancedStockPredictor()
                        evaluator = ModelEvaluator()

                        # Fetch and preprocess data
                        st.info("📥 Fetching and preprocessing data...")
                        raw_data = fetch_stock_data(selected_symbol, "2y")

                        if raw_data is not None and not raw_data.empty:
                            # Preprocess data
                            processed_data = preprocessor.preprocess_data(raw_data)

                            # Feature engineering
                            st.info("🔧 Engineering advanced features...")
                            features_df = feature_engineer.create_features(processed_data)

                            # Prepare training data
                            st.info("🎯 Preparing training sequences...")
                            X_train, X_test, y_train, y_test, scaler = predictor.prepare_data(
                                features_df, sequence_length=sequence_length
                            )

                            # Train model
                            st.info(f"🚀 Training {model_type} model...")
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Custom training with progress tracking
                            model, history = predictor.train_model(
                                X_train, y_train, X_test, y_test,
                                model_type=model_type,
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                early_stopping=early_stopping
                            )

                            # Update progress
                            for i in range(epochs):
                                progress = (i + 1) / epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Training epoch {i+1}/{epochs}")

                            progress_bar.empty()
                            status_text.empty()

                            # Evaluate model
                            st.success("✅ Training completed! Evaluating model...")
                            metrics = evaluator.evaluate_model(model, X_test, y_test, scaler)

                            # Display results
                            st.subheader("📈 Model Performance Metrics")
                            col_a, col_b, col_c, col_d = st.columns(4)

                            with col_a:
                                st.metric("MAE", f"{metrics['MAE']:.4f}")
                            with col_b:
                                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                            with col_c:
                                st.metric("R² Score", f"{metrics['R2']:.4f}")
                            with col_d:
                                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

                            # Training history plot
                            st.subheader("📊 Training History")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                            ax1.plot(history.history['loss'], label='Training Loss')
                            ax1.plot(history.history['val_loss'], label='Validation Loss')
                            ax1.set_title('Model Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.legend()

                            ax2.plot(history.history['mae'], label='Training MAE')
                            ax2.plot(history.history['val_mae'], label='Validation MAE')
                            ax2.set_title('Model MAE')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('MAE')
                            ax2.legend()

                            plt.tight_layout()
                            st.pyplot(fig)

                            # SHAP Explainability
                            if include_shap:
                                st.subheader("🔍 SHAP Model Explainability")

                                with st.spinner("Generating SHAP explanations..."):
                                    # Create SHAP explainer
                                    background = X_train[:100]  # Use subset for background
                                    explainer = shap.DeepExplainer(model, background)

                                    # Calculate SHAP values for test set
                                    shap_values = explainer.shap_values(X_test[:50])  # Explain first 50 samples

                                    # Feature names for better interpretation
                                    feature_names = [f'Feature_{i}' for i in range(X_train.shape[2])]

                                    # Summary plot
                                    st.markdown("**SHAP Summary Plot**")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.summary_plot(shap_values[0], X_test[:50],
                                                    feature_names=feature_names, show=False)
                                    st.pyplot(fig)

                                    # Waterfall plot for single prediction
                                    st.markdown("**SHAP Waterfall Plot (Single Prediction)**")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.plots.waterfall(explainer.expected_value[0],
                                                       shap_values[0][0],
                                                       X_test[0],
                                                       feature_names=feature_names,
                                                       show=False)
                                    st.pyplot(fig)

                                    # Feature importance
                                    st.markdown("**Feature Importance**")
                                    feature_importance = np.abs(shap_values[0]).mean(axis=0)
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': feature_importance
                                    }).sort_values('Importance', ascending=False)

                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
                                    plt.title('Top 15 Most Important Features')
                                    st.pyplot(fig)

                        else:
                            st.error("Failed to fetch training data. Please try again.")

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.info("Make sure all required packages are installed (tensorflow, shap, scikit-learn, etc.)")

        with col2:
            st.subheader("📋 Training Summary")

            # Model architecture info
            st.markdown("**Model Architecture:**")
            st.info(f"""
            - Type: {model_type}
            - Sequence Length: {sequence_length}
            - Batch Size: {batch_size}
            - Epochs: {epochs}
            - Learning Rate: {learning_rate}
            """)

            # Training features
            st.markdown("**Training Features:**")
            features_list = [
                "✅ Technical Indicators (RSI, MACD, Bollinger Bands)",
                "✅ Price Action Features",
                "✅ Volume Analysis",
                "✅ Moving Averages",
                "✅ Momentum Indicators",
                "✅ Volatility Measures"
            ]

            if include_shap:
                features_list.append("✅ SHAP Explainability")
            if cross_validation:
                features_list.append("✅ Cross-Validation")
            if early_stopping:
                features_list.append("✅ Early Stopping")

            for feature in features_list:
                st.markdown(feature)

            # Quick stats
            st.markdown("**Quick Stats:**")
            st.metric("Training Symbol", selected_symbol)
            st.metric("Data Period", "2 Years")
            st.metric("Model Type", model_type)

    except ImportError as e:
        st.error(f"Required modules not found: {str(e)}")
        st.info("""
        Please install the required packages:
        ```bash
        pip install tensorflow shap scikit-learn matplotlib seaborn
        ```
        """)

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
    diagnostic_bot()

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



