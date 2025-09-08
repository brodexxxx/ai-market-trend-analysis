import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf

# Technical indicators function
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
    
    return df

# Investment recommendation function
def generate_recommendation(df, symbol):
    if df is None or df.empty:
        return "⚠️ Insufficient data for recommendation"
    
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    
    recommendation = f"## 📊 Analysis for {symbol}\n\n"
    
    if current_price > sma_50:
        recommendation += "✅ **Bullish Trend** - Price above 50-day SMA\n"
    else:
        recommendation += "⚠️ **Bearish Trend** - Price below 50-day SMA\n"
    
    if rsi < 30:
        recommendation += "📈 **Oversold** - Strong buying opportunity\n"
    elif rsi > 70:
        recommendation += "📉 **Overbought** - Consider profit-taking\n"
    else:
        recommendation += "📊 **Neutral RSI** - Monitor for entry points\n"
    
    return recommendation

# Fetch stock data with fallback
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.info(f"Real-time data for {symbol} is currently unavailable. Using sample data.")
            return generate_sample_data(symbol, period)
            
        return hist
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Using sample data for demonstration.")
        return generate_sample_data(symbol, period)

def generate_sample_data(symbol, period):
    """Generate sample stock data"""
    try:
        if period == '1mo':
            days = 30
        elif period == '3mo':
            days = 90
        elif period == '6mo':
            days = 180
        else:
            days = 365
            
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        np.random.seed(42)
        base_price = 100 + np.random.randint(-20, 20)
        
        prices = []
        current_price = base_price
        for _ in range(days):
            change = np.random.normal(0, 2)
            current_price = max(10, current_price + change)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'Open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in range(days)]
        }, index=dates)
        
        return df
        
    except Exception as e:
        st.error(f"Error generating sample data: {e}")
        return None

# Streamlit app
st.set_page_config(page_title="AI Stock Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stPlotlyChart {
        height: 600px !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Stock Analysis", "Market News"])

if page == "Stock Analysis":
    # Sidebar
    st.sidebar.header("📊 Stock Selection")
    
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
    
    selected_category = st.sidebar.selectbox("Select Stock Category:", list(popular_stocks.keys()), index=0)
    selected_stock = st.sidebar.selectbox("Select Stock:", list(popular_stocks[selected_category].keys()), index=0)
    
    custom_symbol = st.sidebar.text_input("Or enter custom symbol:", "").upper()
    
    if custom_symbol:
        symbol = custom_symbol
    else:
        symbol = popular_stocks[selected_category][selected_stock]
    
    period = st.sidebar.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    st.title("🤖 AI-Powered Stock Market Analysis")
    st.markdown("---")
    
    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner("Fetching and analyzing data..."):
            df = fetch_stock_data(symbol, period)
            
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                
                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🎯 Patterns", "📋 Statistics"])
                
                with tab1:
                    # Price chart
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('Price Chart with Moving Averages', 'Volume Analysis'),
                                      vertical_spacing=0.1,
                                      row_width=[0.3, 0.7])
                    
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    if 'SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['SMA_20'], 
                            name='SMA 20', line=dict(color='orange', width=2)
                        ), row=1, col=1)
                    
                    if 'SMA_50' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['SMA_50'], 
                            name='SMA 50', line=dict(color='purple', width=2)
                        ), row=1, col=1)
                    
                    fig.add_trace(go.Bar(
                        x=df.index, y=df['Volume'], 
                        name='Volume', marker_color='lightblue'
                    ), row=2, col=1)
                    
                    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Technical indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'RSI' in df.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(width=3)))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", line_width=2)
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", line_width=2)
                            fig_rsi.update_layout(title="RSI Indicator", height=400)
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with col2:
                        if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(width=3)))
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=dict(width=3)))
                            fig_macd.update_layout(title="MACD Analysis", height=400)
                            st.plotly_chart(fig_macd, use_container_width=True)
                
                with tab3:
                    # Pattern recognition
                    st.info("Pattern recognition feature coming soon!")
                
                with tab4:
                    # Statistics
                    st.subheader("📊 Key Statistics")
                    
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
                        st.metric("Volatility", f"{df['Close'].pct_change().std()*100:.2f}%")
                    
                    # Recommendation
                    st.subheader("🎯 AI Investment Recommendation")
                    recommendation = generate_recommendation(df, symbol)
                    st.markdown(recommendation)
            
            else:
                st.error("Failed to fetch stock data. Please try again.")

elif page == "Market News":
    st.title("📰 Market News & Insights")
    st.markdown("---")
    
    news_items = [
        {
            "title": "Federal Reserve Signals Potential Rate Cuts",
            "summary": "The Fed indicated it may begin cutting interest rates later this year.",
            "impact": "Positive",
            "date": "2024-01-15"
        },
        {
            "title": "Tech Giants Report Strong Earnings",
            "summary": "Major tech companies reported better-than-expected quarterly results.",
            "impact": "Positive",
            "date": "2024-01-14"
        }
    ]
    
    for news in news_items:
        with st.expander(f"{news['title']} - {news['date']}"):
            st.markdown(f"**Summary:** {news['summary']}")
            st.markdown(f"**Market Impact:** {news['impact']}")

st.markdown("---")
st.caption("📊 Powered by AI Market Trend Analysis | Data from Yahoo Finance")
