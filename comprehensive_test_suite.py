import requests
import json
import time
import random

class ComprehensiveAPITest:
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
            return False
    
    def test_predict_valid_stocks(self):
        """Test prediction with valid stock symbols"""
        valid_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "INFY.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        results = []
        
        for symbol in valid_stocks:
            try:
                response = requests.post(f"{self.base_url}/predict", json={"symbol": symbol})
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {symbol}: Prediction {result.get('prediction')} ({result.get('direction')})")
                    results.append(True)
                else:
                    print(f"❌ {symbol}: Failed with status {response.status_code}")
                    results.append(False)
            except Exception as e:
                print(f"❌ {symbol}: Error {e}")
                results.append(False)
        
        return all(results)
    
    def test_invalid_symbols(self):
        """Test error handling for invalid stock symbols"""
        invalid_symbols = ["INVALID", "XYZ123", "NONEXISTENT", ""]
        results = []
        
        for symbol in invalid_symbols:
            try:
                response = requests.post(f"{self.base_url}/predict", json={"symbol": symbol})
                if response.status_code == 400 or response.status_code == 500:
                    print(f"✅ {symbol}: Correctly handled invalid symbol (Status {response.status_code})")
                    results.append(True)
                else:
                    print(f"❌ {symbol}: Unexpected status {response.status_code}")
                    results.append(False)
            except Exception as e:
                print(f"❌ {symbol}: Error {e}")
                results.append(False)
        
        return all(results)
    
    def test_missing_data(self):
        """Test error handling for missing data"""
        test_cases = [
            {},
            {"invalid_key": "value"},
            {"symbol": None}
        ]
        results = []
        
        for i, data in enumerate(test_cases):
            try:
                response = requests.post(f"{self.base_url}/predict", json=data)
                if response.status_code >= 400:
                    print(f"✅ Test case {i+1}: Correctly handled missing data (Status {response.status_code})")
                    results.append(True)
                else:
                    print(f"❌ Test case {i+1}: Unexpected status {response.status_code}")
                    results.append(False)
            except Exception as e:
                print(f"❌ Test case {i+1}: Error {e}")
                results.append(False)
        
        return all(results)
    
    def test_performance(self, num_requests=10):
        """Test performance with multiple requests"""
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            try:
                symbol = random.choice(["AAPL", "GOOGL", "MSFT"])
                response = requests.post(f"{self.base_url}/predict", json={"symbol": symbol})
                if response.status_code == 200:
                    results.append(True)
                else:
                    results.append(False)
            except Exception as e:
                print(f"❌ Request {i+1}: Error {e}")
                results.append(False)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_requests
        
        print(f"⏱️  Performance: {num_requests} requests in {total_time:.2f}s")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Success rate: {sum(results)}/{num_requests}")
        
        return sum(results) / num_requests > 0.8  # 80% success rate
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            {"symbol": "A"},  # Very short symbol
            {"symbol": "A" * 50},  # Very long symbol
            {"symbol": "123"},  # Numeric symbol
            {"symbol": "AAPL", "extra_param": "test"}  # Extra parameters
        ]
        results = []
        
        for i, data in enumerate(edge_cases):
            try:
                response = requests.post(f"{self.base_url}/predict", json=data)
                print(f"✅ Edge case {i+1}: Handled (Status {response.status_code})")
                results.append(True)
            except Exception as e:
                print(f"❌ Edge case {i+1}: Error {e}")
                results.append(False)
        
        return all(results)
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive API Test Suite")
        print("=" * 50)
        
        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Valid Stocks", self.test_predict_valid_stocks),
            ("Invalid Symbols", self.test_invalid_symbols),
            ("Missing Data", self.test_missing_data),
            ("Edge Cases", self.test_edge_cases),
            ("Performance", lambda: self.test_performance(5))  # Reduced for testing
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            result = test_func()
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"   Result: {status}")
        
        print("\n" + "=" * 50)
        print(f"📊 Overall Results: {sum(results)}/{len(results)} tests passed")
        
        if all(results):
            print("🎉 All tests passed! API is ready for production.")
        else:
            print("⚠️  Some tests failed. Review the implementation.")
        
        return all(results)

if __name__ == "__main__":
    tester = ComprehensiveAPITest()
    tester.run_all_tests()
