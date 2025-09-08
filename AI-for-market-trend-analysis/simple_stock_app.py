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
    
    return df

# Fetch stock data with fallback to sample data
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.info(f"Real-time data for {symbol} is currently unavailable. Using sample data for demonstration.")
            return generate_sample_data(symbol, period)
            
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        st.info("Using sample data for demonstration.")
        return generate_sample_data(symbol, period)

def generate_sample_data(symbol, period):
    """Generate sample stock data for demonstration"""
    try:
        if period == '1mo':
            days = 30
        elif period == '3mo':
            days = 90
        elif period == '6mo':
            days = 180
        else:  # 1y
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
st.set_page_config(page_title="Stock Analysis", layout="wide")

st.title("📊 Stock Market Analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol:", "AAPL").upper()
period = st.sidebar.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.sidebar.button("Analyze Stock"):
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(symbol, period)
        
        if df is not None and not df.empty:
            df = calculate_technical_indicators(df)
            
            # Price chart
            st.subheader(f"{symbol} Price Chart")
            fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
            
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
                    x=df.index, 
                    y=df['SMA_20'], 
                    name='SMA 20', 
                    line=dict(color='orange')
                ), row=1, col=1)
            
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['SMA_50'], 
                    name='SMA 50', 
                    line=dict(color='purple')
                ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=df.index, 
                y=df['Volume'], 
                name='Volume', 
                marker_color='lightblue'
            ), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.subheader("Technical Indicators")
            
            if 'RSI' in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI (14 Period)")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Statistics
            st.subheader("Key Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                if 'SMA_20' in df.columns:
                    st.metric("20-Day SMA", f"${df['SMA_20'].iloc[-1]:.2f}")
                if 'SMA_50' in df.columns:
                    st.metric("50-Day SMA", f"${df['SMA_50'].iloc[-1]:.2f}")
            
            with col2:
                st.metric("Volume (Avg)", f"{df['Volume'].mean():,.0f}")
                if 'RSI' in df.columns:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                st.metric("Volatility", f"{df['Close'].pct_change().std()*100:.2f}%")
        
        else:
            st.error("Failed to fetch stock data. Please try a different symbol.")

st.markdown("---")
st.caption("Powered by AI Market Analysis | Data from Yahoo Finance")
