import requests
import json

# Test the API endpoint
url = "http://127.0.0.1:5001/predict"

# Test data for AAPL stock
test_data = {
    "symbol": "AAPL"
}

try:
    response = requests.post(url, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API Test Successful!")
        print(f"Symbol: {result.get('symbol')}")
        print(f"Prediction: {result.get('prediction')} ({result.get('direction')})")
        print(f"Confidence: {result.get('confidence'):.2f}")
        print(f"Current Price: ${result.get('current_price'):.2f}")
        print(f"Price Change: ${result.get('price_change'):.2f} ({result.get('price_change_pct'):.2f}%)")
        print(f"Recommendation: {result.get('recommendation')}")
        print(f"RSI: {result.get('technical_data', {}).get('rsi', 0):.2f}")
        print(f"SMA 20: ${result.get('technical_data', {}).get('sma_20', 0):.2f}")
        print(f"SMA 50: ${result.get('technical_data', {}).get('sma_50', 0):.2f}")
    else:
        print(f"❌ API Test Failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Make sure the Flask server is running on http://127.0.0.1:5001")
except Exception as e:
    print(f"❌ Error: {e}")
