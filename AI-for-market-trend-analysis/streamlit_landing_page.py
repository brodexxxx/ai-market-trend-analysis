import streamlit as st

st.set_page_config(page_title="Positivus Landing Page", layout="wide")

# Header Section
st.title("Welcome to Positivus")
st.markdown("### Your AI-Powered Stock Market Analysis Platform")
st.image("path_to_your_image.jpg")  # Replace with actual image path

# Call to Action
st.markdown("#### Get Started with Our Features")
st.button("Explore Features")

# Features Section
st.markdown("### Features")
st.markdown("- **Real-time Stock Data**: Get the latest stock prices.")
st.markdown("- **AI Predictions**: Leverage AI for stock market predictions.")
st.markdown("- **Interactive Charts**: Visualize stock trends with interactive charts.")

# Footer
st.markdown("---")
st.caption("📊 Powered by AI Market Trend Analysis | Real-time data from Yahoo Finance")
