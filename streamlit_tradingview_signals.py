import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from datetime import datetime
from tradingview_ta import TA_Handler, Interval
import yfinance as yf

# -------- CONFIG --------
INDIAN = [
    "NSE:TCS", "NSE:RELIANCE", "NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:INFY",
    "NSE:ITC", "NSE:LT", "NSE:SBIN", "NSE:BHARTIARTL", "NSE:KOTAKBANK",
    "NSE:AXISBANK", "NSE:MARUTI", "NSE:BAJFINANCE", "NSE:ASIANPAINT", "NSE:WIPRO",
    "NSE:HCLTECH", "NSE:TITAN", "NSE:HINDUNILVR", "NSE:ULTRACEMCO", "NSE:NTPC",
    "NSE:POWERGRID", "NSE:ADANIENT", "NSE:ONGC", "NSE:BAJAJFINSV", "NSE:JSWSTEEL"
]
GLOBAL = [
    "NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:AMZN", "NASDAQ:NVDA",
    "NYSE:BRK.B", "NASDAQ:META", "NASDAQ:TSLA", "NYSE:JNJ", "NYSE:V",
    "NYSE:MA", "NYSE:JPM", "NYSE:PG", "NASDAQ:AVGO", "NYSE:UNH",
    "NYSE:HD", "NYSE:KO", "NYSE:PEP", "NASDAQ:ADBE", "NASDAQ:COST",
    "NYSE:PFE", "NYSE:XOM", "NYSE:CVX", "NASDAQ:INTC", "NASDAQ:CSCO"
]
TIMEFRAME = Interval.INTERVAL_1_DAY
PAUSE = 1.0
# ------------------------

def trendlyne_stub(symbol: str) -> dict:
    # Placeholder: Trendlyne does not publish public dev docs; integrate when B2B API contract is available.
    # Return empty or cached business metrics if privately provisioned.
    return {"trendlyne_note": "Integrate when API access is available."}

def tv_summary(symbol: str):
    # Input validation
    if not symbol or ":" not in symbol:
        raise ValueError("Invalid symbol format")
    exchange, ticker = symbol.split(":", 1)
    if exchange not in ["NSE", "BSE", "NASDAQ", "NYSE"]:
        raise ValueError("Unsupported exchange")
    if not ticker or len(ticker) > 10:
        raise ValueError("Invalid ticker")

    screener = "india" if exchange in ("NSE", "BSE") else "america"
    try:
        h = TA_Handler(symbol=ticker, screener=screener, exchange=exchange, interval=TIMEFRAME)
        a = h.get_analysis()

        # Handle different response formats
        if isinstance(a, dict):
            s = a.get("summary", {})
            osc = a.get("oscillators", {})
            ma = a.get("moving_averages", {})
        else:
            s = a.summary
            osc = a.oscillators
            ma = a.moving_averages

        reasons = [
            f"Oscillators: {osc.get('RECOMMENDATION', 'NA')} "
            f"({osc.get('BUY', 0)} buy/{osc.get('NEUTRAL', 0)} neutral/{osc.get('SELL', 0)} sell)",
            f"MAs: {ma.get('RECOMMENDATION', 'NA')} "
            f"({ma.get('BUY', 0)} buy/{ma.get('NEUTRAL', 0)} neutral/{ma.get('SELL', 0)} sell)"
        ]
        for label, comp in (("MA", ma.get("COMPUTE", {})), ("OSC", osc.get("COMPUTE", {}))):
            bulls = [k for k, v in comp.items() if v == "BUY"]
            bears = [k for k, v in comp.items() if v == "SELL"]
            if bulls: reasons.append(f"Bullish {label}: {', '.join(bulls[:6])}")
            if bears: reasons.append(f"Bearish {label}: {', '.join(bears[:6])}")
        chart = f"https://in.tradingview.com/chart/?symbol={exchange}%3A{ticker}"
        return {
            "recommendation": s.get("RECOMMENDATION", "NEUTRAL"),
            "buy": s.get("BUY", 0),
            "neutral": s.get("NEUTRAL", 0),
            "sell": s.get("SELL", 0),
            "reasons": reasons,
            "chart": chart
        }
    except Exception as e:
        # Don't show error in UI, just return error data
        return {
            "recommendation": "ERROR",
            "buy": 0,
            "neutral": 0,
            "sell": 0,
            "reasons": ["Data fetch failed"],
            "chart": ""
        }

@st.cache_data(ttl=300)  # Cache for 5 minutes
def tv_hist(symbol: str) -> pd.DataFrame:
    exch, tick = symbol.split(":")
    try:
        # Convert to yfinance format
        if exch == "NSE":
            yf_symbol = f"{tick}.NS"
        elif exch == "BSE":
            yf_symbol = f"{tick}.BO"
        elif exch == "NASDAQ" or exch == "NYSE":
            yf_symbol = tick
        else:
            yf_symbol = f"{tick}.{exch}"

        stock = yf.Ticker(yf_symbol)
        df = stock.history(period="2y")  # Get 2 years of data
        if df.empty:
            # If yfinance fails, create mock data for demo purposes
            return create_mock_data(symbol)
        df = df.reset_index().rename(columns={"Date": "date"})
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        # If yfinance fails, create mock data for demo purposes
        # Suppress the error message to avoid cluttering the console
        return create_mock_data(symbol)

def create_mock_data(symbol: str) -> pd.DataFrame:
    """Create mock historical data when yfinance fails"""
    import numpy as np
    from datetime import datetime, timedelta

    # Generate 500 days of mock data
    dates = [datetime.now() - timedelta(days=i) for i in range(500, 0, -1)]

    # Generate realistic price movements
    np.random.seed(hash(symbol) % 2**32)  # Deterministic seed based on symbol
    base_price = 1000 if "NSE:" in symbol else 200  # Different base prices for different markets

    prices = [base_price]
    for i in range(499):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure positive prices

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(100000, 10000000) for _ in prices]
    })

    df = df.sort_values('date').reset_index(drop=True)
    return df

def enrich_trend(df: pd.DataFrame) -> dict:
    out = {"ma50": None, "ma200": None, "bias": "Unknown", "rsi14": None}
    if df.empty or "close" not in df:
        return out
    close = df["close"].astype(float)
    out["ma50"] = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
    out["ma200"] = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
    if out["ma50"] and out["ma200"]:
        out["bias"] = "Uptrend" if out["ma50"] > out["ma200"] else ("Downtrend" if out["ma50"] < out["ma200"] else "Range")
    # Simple RSI14
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = np.where(down == 0, np.nan, up / down)
    rsi = 100 - (100 / (1 + rs))
    out["rsi14"] = float(pd.Series(rsi).iloc[-1])
    return out

def build_row(symbol: str):
    base = tv_summary(symbol)
    hist = tv_hist(symbol)
    t = enrich_trend(hist)
    trn = trendlyne_stub(symbol)
    why = base["reasons"] + [
        f"Trend bias: {t['bias']} (MA50 {t['ma50']:.2f} vs MA200 {t['ma200']:.2f})" if t["ma50"] and t["ma200"] else "Trend bias: insufficient data",
        f"RSI14: {t['rsi14']:.1f}" if t["rsi14"] else "RSI14: n/a"
    ]
    if trn.get("trendlyne_note"):
        why.append(trn["trendlyne_note"])
    # AI-style decision explanation
    decision = base["recommendation"]
    hold_reason = "Hold rationale: positive trend with neutral oscillators" if (t["bias"] == "Uptrend" and base["neutral"] >= base["sell"]) else "Hold rationale: mixed signals; await confirmation"
    return {
        "symbol": symbol,
        "region": "India" if symbol.startswith("NSE:") or symbol.startswith("BSE:") else "International",
        "decision": decision,
        "buy": base["buy"],
        "neutral": base["neutral"],
        "sell": base["sell"],
        "why": " | ".join(why),
        "hold_reason": hold_reason,
        "chart": base["chart"]
    }, hist

@st.cache_data(ttl=300)
def generate_plots(sample_series: list):
    # Pick up to 5 symbols across regions to plot trends
    picks = (INDIAN[:2] + GLOBAL[:3])[:5]
    images = {}
    for sym in picks:
        df = tv_hist(sym)
        if df.empty: continue
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df["date"], df["close"], label="Close", color="#1f77b4")
        if df["ma50"].notna().any(): ax.plot(df["date"], df["ma50"], label="MA50", color="#ff7f0e")
        if df["ma200"].notna().any(): ax.plot(df["date"], df["ma200"], label="MA200", color="#2ca02c")
        ax.set_title(f"{sym} daily trend")
        ax.legend()
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        images[sym] = buf
    return images

def main():
    st.set_page_config(page_title="AI for Market Trend Analysis", layout="wide")
    st.title("📊 AI for Market Trend Analysis Dashboard")

    # Warning about data source
    st.warning("⚠️ **Note:** Due to current API limitations with Indian stocks, the app may use simulated data for demonstration purposes. Real-time data fetching is attempted first, with fallback to mock data when necessary.")

    # Sidebar controls
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (minutes)", 1, 60, 5) if auto_refresh else 5

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Signals", "Charts", "Q&A", "Settings"])

    with tab1:
        st.header("Stock Signals")

        # Create columns for separate buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Compute Indian Signals", type="primary"):
                with st.spinner("Computing Indian stock signals..."):
                    rows = []
                    for sym in INDIAN:
                        row, _ = build_row(sym)
                        rows.append(row)
                        time.sleep(PAUSE)
                    df = pd.DataFrame(rows)
                    st.session_state.signals_df = df
                    st.success(f"Computed signals for {len(df)} Indian stocks")

        with col2:
            if st.button("Compute International Signals", type="primary"):
                with st.spinner("Computing international stock signals..."):
                    rows = []
                    for sym in GLOBAL:
                        row, _ = build_row(sym)
                        rows.append(row)
                        time.sleep(PAUSE)
                    df = pd.DataFrame(rows)
                    st.session_state.signals_df = df
                    st.success(f"Computed signals for {len(df)} international stocks")

        with col3:
            if st.button("Compute All Signals", type="secondary"):
                with st.spinner("Computing all stock signals..."):
                    rows = []
                    for sym in INDIAN + GLOBAL:
                        row, _ = build_row(sym)
                        rows.append(row)
                        time.sleep(PAUSE)
                    df = pd.DataFrame(rows)
                    st.session_state.signals_df = df
                    st.success(f"Computed signals for {len(df)} stocks")

        if 'signals_df' in st.session_state:
            df = st.session_state.signals_df
            st.dataframe(df, width='stretch')

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                region_filter = st.selectbox("Filter by Region", ["All", "India", "International"])
            with col2:
                decision_filter = st.selectbox("Filter by Decision", ["All"] + list(df["decision"].unique()))

            filtered_df = df.copy()
            if region_filter != "All":
                filtered_df = filtered_df[filtered_df["region"] == region_filter]
            if decision_filter != "All":
                filtered_df = filtered_df[filtered_df["decision"] == decision_filter]

            st.subheader("Filtered Results")
            st.dataframe(filtered_df, width='stretch')

            # Show prediction accuracy prominently
            from security_bot import security_bot
            accuracy = security_bot.get_prediction_accuracy()
            st.markdown(f"### Current Model Prediction Accuracy: **{accuracy}**")

            # Show detailed POV for each stock in a large box (~5cm height)
            st.markdown("### Stock Point of View Details")
            for idx, row in df.iterrows():
                pov = security_bot.get_stock_pov_details(row['symbol'], {
                    "recommendation": row['decision'],
                    "buy": row['buy'],
                    "neutral": row['neutral'],
                    "sell": row['sell'],
                    "reasons": row['why'].split(" | ") if 'why' in row else []
                })
                with st.expander(f"{row['symbol']} - {pov['recommendation']}"):
                    st.write(f"Confidence Score: {pov['confidence_score']:.2f}")
                    st.write(f"Technical Analysis: {pov['technical_analysis']}")
                    st.write(f"Risk Assessment: {pov['risk_assessment']}")
                    st.write(f"Trading Strategy: {pov['trading_strategy']}")
                    st.write(f"Support/Resistance Levels: {pov['next_support_resistance']}")
                    st.write(f"Market Sentiment: {pov['market_sentiment']}")
                    st.write(f"Time Horizon: {pov['time_horizon']}")
                    st.write(f"Position Sizing: {pov['position_sizing']}")
                    st.write(f"Stop Loss/Target: {pov['stop_loss_target']}")

    with tab2:
        st.header("Charts")

        # Create columns for region selection
        col1, col2 = st.columns(2)
        with col1:
            region_choice = st.radio("Select Region", ["Indian Stocks", "International Stocks"], horizontal=True)

        # Filter symbols based on region choice
        if region_choice == "Indian Stocks":
            available_symbols = INDIAN
        else:
            available_symbols = GLOBAL

        symbol = st.selectbox("Select Symbol for Chart", available_symbols)

        if st.button("Generate Chart"):
            with st.spinner("Generating chart..."):
                df = tv_hist(symbol)
                if not df.empty:
                    df["ma50"] = df["close"].rolling(50).mean()
                    df["ma200"] = df["close"].rolling(200).mean()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df["date"], df["close"], label="Close", color="#1f77b4")
                    if df["ma50"].notna().any(): ax.plot(df["date"], df["ma50"], label="MA50", color="#ff7f0e")
                    if df["ma200"].notna().any(): ax.plot(df["date"], df["ma200"], label="MA200", color="#2ca02c")
                    ax.set_title(f"{symbol} Daily Trend")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.error("No data available for this symbol")

    with tab3:
        st.header("Q&A")
        question = st.text_input("Ask a question about the signals:")
        if st.button("Ask"):
            if 'signals_df' not in st.session_state:
                st.error("Please compute signals first")
            else:
                df = st.session_state.signals_df
                q = question.lower()
                if "strong buy" in q or ("best" in q and "buy" in q):
                    out = df.sort_values(["buy", "sell"], ascending=[False, True]).head(5)
                    st.write("Top 5 by buy-count:")
                    st.dataframe(out)
                elif "hold" in q:
                    out = df[df["decision"].str.contains("HOLD", case=False, na=False)].head(10)
                    st.write("Sample HOLD signals:")
                    st.dataframe(out)
                elif "india" in q:
                    out = df[df["region"] == "India"].head(10)
                    st.write("Sample India rows:")
                    st.dataframe(out)
                elif "international" in q or "global" in q:
                    out = df[df["region"] == "International"].head(10)
                    st.write("Sample International rows:")
                    st.dataframe(out)
                else:
                    st.write("Ask e.g., 'top strong buy', 'show holds', 'india', 'international'.")

    with tab4:
        st.header("Settings")
        st.write("Configure your TradingView credentials:")
        tv_user = st.text_input("TV Username", type="password")
        tv_pass = st.text_input("TV Password", type="password")
        if st.button("Save Credentials"):
            st.session_state.tv_user = tv_user
            st.session_state.tv_pass = tv_pass
            st.success("Credentials saved (session only)")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval * 60)
        st.rerun()

if __name__ == "__main__":
    main()
