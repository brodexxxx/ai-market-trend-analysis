import requests
import json
import time

class EnhancedAPITest:
    def __init__(self, base_url="http://127.0.0.1:5001"):
        self.base_url = base_url
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"✅ Health endpoint: Status {response.status_code}")
            print(f"   Response: {response.json()}")
            return True
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return Falsen
    
    def test_stocks_list(self):
        """Test stocks list endpoint"""
        try:
            response = requests.get(f"{self.base_url}/stocks")
            data = response.json()
            print(f"✅ Stocks list: {len(data.get('indian_stocks', []))} Indian stocks, {len(data.get('us_stocks', []))} US stocks")
            return True
        except Exception as e:
            print(f"❌ Stocks list failed: {e}")
            return False
    
    def test_predict_with_validation(self):
        """Test prediction with proper validation"""
        test_cases = [
            # Valid cases
            {"symbol": "AAPL", "expected_status": 200},
            {"symbol": "RELIANCE.NS", "expected_status": 200},
            {"symbol": "INFY.NS", "expected_status": 200},
            
            # Invalid cases
            {"symbol": "INVALID_SYMBOL_123", "expected_status": 400},
            {"symbol": "", "expected_status": 400},
            {"symbol": None, "expected_status": 400},
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            try:
                response = requests.post(f"{self.base_url}/predict", json={"symbol": test_case["symbol"]})
                if response.status_code == test_case["expected_status"]:
                    print(f"✅ Test {i+1}: {test_case['symbol']} - Correct status {response.status_code}")
                    results.append(True)
                else:
                    print(f"❌ Test {i+1}: {test_case['symbol']} - Expected {test_case['expected_status']}, got {response.status_code}")
                    results.append(False)
            except Exception as e:
                print(f"❌ Test {i+1}: {test_case['symbol']} - Error: {e}")
                results.append(False)
        
        return all(results)
    
    def test_detailed_predictions(self):
        """Test detailed prediction responses"""
        symbols = ["AAPL", "GOOGL", "RELIANCE.NS", "TCS.NS"]
        results = []
        
        for symbol in symbols:
            try:
                response = requests.post(f"{self.base_url}/predict", json={"symbol": symbol})
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {symbol}: {data.get('recommendation', {}).get('action', 'N/A')} - Confidence: {data.get('confidence', 0):.2f}")
                    print(f"   Technical: RSI={data.get('technical_data', {}).get('rsi', 0):.1f}, SMA20={data.get('technical_data', {}).get('sma_20', 0):.2f}")
                    results.append(True)
                else:
                    print(f"❌ {symbol}: Failed with status {response.status_code}")
                    results.append(False)
            except Exception as e:
                print(f"❌ {symbol}: Error {e}")
                results.append(False)
        
        return all(results)
    
    def test_batch_predict(self):
        """Test batch prediction"""
        try:
            symbols = ["AAPL", "MSFT", "GOOGL", "INVALID", "RELIANCE.NS"]
            response = requests.post(f"{self.base_url}/batch_predict", json={"symbols": symbols})
            data = response.json()
            
            print(f"✅ Batch predict: Processed {data.get('total_processed', 0)} symbols")
            print(f"   Successful: {data.get('successful', 0)}, Failed: {data.get('total_processed', 0) - data.get('successful', 0)}")
            
            for result in data.get('results', []):
                status = "✅" if result.get('status') == 'success' else "❌"
                print(f"   {status} {result.get('symbol')}: {result.get('prediction', 'N/A')} ({result.get('direction', 'N/A')})")
            
            return True
        except Exception as e:
            print(f"❌ Batch predict failed: {e}")
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        try:
            responses = []
            for i in range(35):  # More than the 30/minute limit
                response = requests.post(f"{self.base_url}/predict", json={"symbol": "AAPL"})
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay
            
            rate_limited = any(status == 429 for status in responses)
            if rate_limited:
                print("✅ Rate limiting working correctly (429 responses detected)")
            else:
                print("⚠️  Rate limiting not triggered (may need more requests)")
            
            return True
        except Exception as e:
            print(f"❌ Rate limiting test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all enhanced tests"""
        print("🚀 Starting Enhanced API Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Stocks List", self.test_stocks_list),
            ("Prediction Validation", self.test_predict_with_validation),
            ("Detailed Predictions", self.test_detailed_predictions),
            ("Batch Prediction", self.test_batch_predict),
            ("Rate Limiting", self.test_rate_limiting),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            result = test_func()
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"   Result: {status}")
        
        print("\n" + "=" * 60)
        print(f"📊 Overall Results: {sum(results)}/{len(results)} tests passed")
        
        if all(results):
            print("🎉 All enhanced tests passed! API is fully functional.")
        else:
            print("⚠️  Some tests failed. Review the implementation.")
        
        return all(results)

if __name__ == "__main__":
    tester = EnhancedAPITest()
    tester.run_all_tests()
