import requests
import json

def test_connection():
    url = "http://127.0.0.1:5001/predict"
    test_data = {"symbol": "AAPL"}
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server is not running on port 5001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
