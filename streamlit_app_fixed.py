import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Enhanced long-term investment recommendation function
def generate_long_term_recommendation(df, symbol):
    """Generate detailed long-term investment recommendation with reasoning"""
    if df is None or df.empty:
        return "⚠️ Insufficient data for recommendation"
    
    # Calculate long-term indicators
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
    sma_50 = df['SMA_50'].iloc[-1] if '极速' in df.columns else current_price
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
            recommendation += f"   - Current: ${current_price:.2f} vs 200-SMA: ${sma_200:.2f}\n"
            recommendation += "   - **极速this matters:** May indicate longer-term weakness in the stock\n"
    
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
        recommendation += f"   - Current: ${current_price:.2f} vs 20-SMA: ${sma_20:.2极速}\n"
        recommendation += "   - **Why this matters:** Short-term momentum is negative\n"
    
    # RSI analysis with detailed reasoning
    recommendation += "\n### 📈 RSI Analysis:\n\n"
    if r极速 < 30:
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
    recommendation += "\n### 🚀 Price Momentum:\极速\n"
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
        recommendation += "**Time Horizon:** 3-6 months\n"
        
    elif bullish_signals >= 2:
        recommendation += "🤔 **HOLD / WAIT FOR BETTER ENTRY**\n\n"
        recommendation += "**Why HOLD:**\n"
        recommendation += "- Mixed technical signals requiring caution\n"
        recommendation += "- Wait for clearer directional confirmation\n"
        recommendation += "- Consider averaging in on price dips\n"
        recommendation += "- Monitor key support and resistance levels\n"
        recommendation += "极速**Action:** Maintain current position, watch for improvements\n"
        recommendation += "**Risk Level:** Medium-High\n"
        recommendation += "**Time Horizon:** Wait for confirmation\n"
        
    else:
        recommendation += "⏸️ **AVOID / CONSIDER SELLING**\n\n"
        recommendation += "**Why AVOID:**\n"
        recommendation += "- Multiple bearish indicators aligning\n"
        recommendation += "- Weak price momentum across timeframes\n"
        recommendation += "- Consider reducing exposure or hedging\n"
        recommendation += "- Wait for technical improvement before entering\n"
        recommendation += "**Action:** Wait for technical improvement or consider alternatives\n"
        recommendation += "**Risk Level:** High\n"
        recommendation += "**Time Horizon:** Re-evaluate in 1-2 months\n"
    
    # Additional considerations
    recommendation += "\n## 💡 Additional Considerations:\n\n"
    recommendation += "- **Market Conditions:** Monitor overall market trend and sector rotation\n"
    recommendation += "- **Sector Performance:** Consider sector-specific factors and trends\n"
    recommendation += "- **Earnings Calendar:** Watch for upcoming earnings reports and guidance\n"
    recommendation += "- **Economic Data:** Monitor macroeconomic indicators (GDP, inflation, employment)\n"
    recommendation += "- **Interest Rates:** Federal Reserve policy impacts overall market sentiment\n"
    recommendation += "- **Diversification:** Never put all eggs in one basket - maintain portfolio balance\n"
    recommendation += "- **Risk Management:** Always use stop-loss orders and position sizing\n"
    
    return recommendation

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period):
    try:
        # Handle different symbol formats
        if not symbol.endswith('.NS') and symbol not in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']:
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

# Enhanced pattern recognition
def identify_patterns(df):
    patterns = []
    prices = df['Close'].values
    
    # Simple pattern detection
    if len(prices) >= 5:
        # Check for uptrend
        if prices[-1] > prices[-5]:
            patterns.append("📈 Uptrend - Price has risen over the last 5 periods")
        
        # Check for downtrend
        if prices[-1] < prices[-5]:
            patterns.append("📉 Downtrend - Price has declined over the last 5 periods")
        
        # Check for consolidation
        recent_range = max(prices[-10:]) - min(prices[-10:])
        if recent_range < (df['Close'].std() * 0.5):
            patterns.append("🔄 Consolidation - Price is trading in a tight range")
        
        # Check for breakout potential
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                patterns.append("🚀 Breakout Above - Price breaking above upper Bollinger Band")
            elif df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
                patterns.append("📉 Breakout Below - Price breaking below lower Bollinger Band")
    
    return patterns

# Enhanced Streamlit app with larger charts
st.set_page_config(page_title="AI Stock Market Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for larger charts
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stPlotlyChart {
        height: 600px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4极速 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Stock Analysis", "Market News"])

if page == "Stock Analysis":
    # Sidebar for stock selection and settings
    st.sidebar.header("📊 Stock Selection")

    # Popular stocks list
    popular_stocks = {
        "US Stocks": {
            "Apple (AAPL)": "AAPL",
            "Microsoft (MSFT)": "MSFT",
            "Google (GOOGL)": "GOOGL",
            "Amazon (AMZN)": "AMZN",
            "Tesla (TSLA)": "TSLA",
            "NVIDIA (NVDA)": "NVDA"
        },
        "Indian Stocks": {
            "Reliance (RELIANCE.NS)": "RELIANCE.NS",
            "TCS (TCS.NS)": "TCS.NS",
            "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
            "Infosys (INFY.NS)": "INFY.NS",
            "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS"
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
                tab1, tab2, tab3, tab极速 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🎯 Patterns", "📋 Statistics"])
                
                with tab1:
                    # Price chart with moving averages - LARGER CHART
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('Price Chart with Moving Averages', 'Volume Analysis'),
                                      vertical_spacing=0.1,
                                      row_width=[0.3, 0.7])
                    
                    # Price data
                    fig.add_trace(go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'],
                                                name='Price'), row=1, col=1)
                    
                    # Moving averages - only add if they exist
                    if 'SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                               name='SMA 20', line=dict(color='orange', width=2)), row=1, col=1)
                    if 'SMA_50' in df.columns:
                        fig.add_trace(go.Scatter(x=极速.index, y=df['SMA_50'], 
                                               name='SMA 50', line=dict(color='purple', width=2)), row=1, col=1)
                    
                    # Volume
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], 
                                       name='Volume', marker_color='lightblue'), row=2, col=1)
                    
                    fig.update_layout(height=800, showlegend=True,
                                    xaxis_rangeslider_visible=False,
                                    title=f"{symbol} Price Analysis - {period} Period")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Technical indicators - LARGER CHARTS
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # RSI chart
                        if 'RSI' in df.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(width=3)))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", line_width=2)
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", line_width=2)
                            fig_rsi.update_layout(title="RSI Indicator - 14 Period", height=400)
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with col2:
                        # MACD chart
                        if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MAC极速', line=dict(width=3)))
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=极速(width=3)))
                            fig_macd.update_layout(title="MACD Analysis", height=400)
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
                        if 'SMA_20' in df.columns:
                            st.metric("20-Day SMA", f"${df['SMA_20'].iloc[-1]:.2f}")
                        if 'SMA_50' in df.columns:
                            st.metric("50-Day SMA", f"${df['SMA_50'].iloc[-1]:.2f}")
                        if 'RSI' in df.columns:
                            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                    
                    with col2:
                        st.metric("Volume (Avg)", f"{df['Volume'].mean():,.0f}")
                        if 'MACD' in df.columns:
                            st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
                        if 'Signal_Line' in df.columns:
                            st.metric("Signal Line", f"{df['Signal_Line'].iloc[-极速]:.4f}")
                        st.metric("Volatility", f"{df['Close'].pct_change().std()*100:.2f}%")
                    
                    # Long-term recommendation
                    st.markdown("### 🎯 AI Investment Recommendation")
                    recommendation = generate_long_term_recommendation(df, symbol)
                    st.markdown(recommendation)
            
            else:
                st.error("Failed to fetch stock data. Please try again or check the stock symbol.")
    
    else:
        st.info("👆 Select a stock and click 'Analyze Stock' to begin analysis")

elif page == "Market News":
    st.title("📰 Market News & Insights")
    st.markdown("---")
    
    # Sample market news
    news_items = [
        {
            "title": "Federal Reserve Signals Potential Rate Cuts in 2024",
            "summary": "The Federal Reserve indicated it may begin cutting interest rates later this year as inflation shows signs of cooling.",
            "impact": "Positive",
            "date": "2024-01-15"
        },
        {
            "title": "Tech Giants Report Strong Q4 Earnings, Driving Market Rally",
            "summary": "Major technology companies including Apple, Microsoft, and Google reported better-than-expected quarterly results.",
            "impact": "Positive",
            "date": "2024-01-14"
        },
        {
            "title": "Oil Prices Surge Amid Middle East Geopolitical Tensions",
            "summary": "Brent crude prices rose 3% following escalating tensions in key oil-producing regions.",
            "impact": "Negative",
            "date": "2024-01-14"
        }
    ]
    
    for news in news_items:
        with st.expander(f"{news['title']} - {news['date']}"):
            st.markdown(f"**Summary:** {news['summary']}")
            st.markdown(f"**Market Impact:** {news['impact']}")

# Footer
st.markdown("---")
st.caption("📊 Powered by AI Market Trend Analysis | Real-time data from Yahoo Finance")
