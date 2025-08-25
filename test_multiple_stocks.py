import requests
import json

def test_stock_prediction(symbol):
    """Test prediction for a specific stock symbol"""
    url = "http://127.0.0.1:5001/predict"
    test_data = {"symbol": symbol}
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📈 {symbol} Prediction Results:")
            print(f"   Prediction: {result.get('prediction')} ({result.get('direction')})")
            print(f"   Confidence: {result.get('confidence'):.2f}")
            print(f"   Current Price: ${result.get('current_price'):.2f}")
            print(f"   Price Change: ${result.get('price_change'):.2f} ({result.get('price_change_pct'):.2f}%)")
            print(f"   Recommendation: {result.get('recommendation')}")
            print(f"   RSI: {result.get('technical_data', {}).get('rsi', 0):.2f}")
            return True
        else:
            print(f"❌ {symbol} Test Failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {symbol}: {e}")
        return False

if __name__ == "__main__":
    print("Testing Stock Prediction API with multiple symbols...")
    
    # Test multiple popular stocks
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    for stock in stocks:
        test_stock_prediction(stock)
