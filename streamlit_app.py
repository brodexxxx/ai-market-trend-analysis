import streamlit as st
import requests

st.title("Stock Market Prediction App")

# User input for stock symbol
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
days = st.number_input("Days for prediction:", min_value=1, max_value=365, value=30)

if st.button("Predict"):
    # Call the Flask API for prediction
    response = requests.post("http://127.0.0.1:5001/predict", json={"symbol": symbol, "days": days})
    
    if response.status_code == 200:
        prediction_data = response.json()
        st.success(f"Prediction: {prediction_data['prediction']}, Confidence: {prediction_data['confidence']}")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")

st.sidebar.header("Available Stocks")
st.sidebar.write("Indian Stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS, ...")
st.sidebar.write("US Stocks: AAPL, MSFT, GOOGL, ...")
