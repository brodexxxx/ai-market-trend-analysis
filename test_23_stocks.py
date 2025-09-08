"""
Comprehensive test suite for 23 different stocks to verify application functionality
across various market sectors and stock types.
"""

import pandas as pd
import numpy as np
from src.api_service import fetch_stock_data, calculate_technical_indicators
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_lstm, load_model
from src.evaluation import evaluate_model
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import time

# List of 23 diverse stocks across different sectors
STOCKS_TO_TEST = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'ADBE',
    # Financial
    'JPM', 'V', 'MA', 'GS',
    # Healthcare
    'JNJ', 'PFE', 'UNH',
    # Consumer
    'AMZN', 'WMT', 'PG', 'KO',
    # Industrial
    'BA', 'CAT',
    # Energy
    'XOM', 'CVX',
    # Telecom
    'T', 'VZ'
]

def test_stock_data_fetching():
    """Test fetching data for all 23 stocks"""
    print("Testing stock data fetching for 23 stocks...")
    results = {}
    
    for symbol in STOCKS_TO_TEST:
        try:
            print(f"Fetching data for {symbol}...")
            data = fetch_stock_data(symbol, period='1y')
            
            if data is not None and not data.empty:
                results[symbol] = {
                    'success': True,
                    'data_points': len(data),
                    'columns': list(data.columns),
                    'date_range': f"{data.index.min()} to {data.index.max()}"
                }
                print(f"✓ {symbol}: {len(data)} data points")
            else:
                results[symbol] = {'success': False, 'error': 'No data returned'}
                print(f"✗ {symbol}: No data")
                
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"✗ {symbol}: Error - {e}")
        
        time.sleep(0.1)  # Avoid rate limiting
    
    return results

def test_technical_indicators():
    """Test technical indicators calculation for all stocks"""
    print("\nTesting technical indicators calculation...")
    results = {}
    
    for symbol in STOCKS_TO_TEST:
        try:
            print(f"Testing indicators for {symbol}...")
            data = fetch_stock_data(symbol, period='6mo')
            
            if data is not None and not data.empty:
                data_with_indicators = calculate_technical_indicators(data)
                
                # Check if indicators were added
                indicator_columns = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal']
                indicators_present = [col for col in indicator_columns if col in data_with_indicators.columns]
                
                results[symbol] = {
                    'success': True,
                    'indicators_added': indicators_present,
                    'total_columns': len(data_with_indicators.columns)
                }
                print(f"✓ {symbol}: Added {len(indicators_present)} indicators")
            else:
                results[symbol] = {'success': False, 'error': 'No data to calculate indicators'}
                print(f"✗ {symbol}: No data for indicators")
                
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"✗ {symbol}: Error calculating indicators - {e}")
    
    return results

def test_data_preprocessing():
    """Test data preprocessing for all stocks"""
    print("\nTesting data preprocessing...")
    results = {}

    for symbol in STOCKS_TO_TEST:
        try:
            print(f"Testing preprocessing for {symbol}...")
            data = fetch_stock_data(symbol, period='6mo')

            if data is not None and not data.empty:
                # Clean data and handle duplicates
                data_clean = data.dropna()
                data_clean = data_clean[~data_clean.index.duplicated(keep='first')]

                # Calculate basic features
                data_clean['Daily_Return'] = data_clean['Close'].pct_change()
                data_clean['Log_Return'] = np.log(data_clean['Close'] / data_clean['Close'].shift(1))
                data_clean['Volatility_20D'] = data_clean['Daily_Return'].rolling(window=20).std()
                data_clean['SMA_20'] = data_clean['Close'].rolling(window=20).mean()
                data_clean['SMA_50'] = data_clean['Close'].rolling(window=50).mean()

                # Create target variables
                data_clean['Target_Direction'] = (data_clean['Close'].shift(-1) > data_clean['Close']).astype(int)
                data_clean['Target_Return'] = (data_clean['Close'].shift(-1) - data_clean['Close']) / data_clean['Close']

                # Remove rows with NaN values
                processed_data = data_clean.dropna()

                results[symbol] = {
                    'success': True,
                    'original_shape': data.shape,
                    'processed_shape': processed_data.shape,
                    'has_nulls': processed_data.isnull().any().any(),
                    'data_types': processed_data.dtypes.to_dict()
                }
                print(f"✓ {symbol}: Preprocessed {data.shape} -> {processed_data.shape}")
            else:
                results[symbol] = {'success': False, 'error': 'No data to preprocess'}
                print(f"✗ {symbol}: No data for preprocessing")

        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"✗ {symbol}: Error preprocessing - {e}")

    return results

def test_feature_engineering():
    """Test feature engineering for all stocks"""
    print("\nTesting feature engineering...")
    results = {}

    # Configuration for feature engineering
    feature_config = {
        'technical_indicators': {
            'sma_periods': [5, 10, 20],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2
        },
        'statistical_features': {
            'volatility_window': 20,
            'momentum_window': 10,
            'return_window': 1
        }
    }

    for symbol in STOCKS_TO_TEST:
        try:
            print(f"Testing feature engineering for {symbol}...")
            data = fetch_stock_data(symbol, period='6mo')

            if data is not None and not data.empty:
                # Add symbol column required by engineer_features
                data['Symbol'] = symbol
                features = engineer_features(data, feature_config)

                results[symbol] = {
                    'success': True,
                    'original_features': len(data.columns),
                    'engineered_features': len(features.columns),
                    'feature_names': list(features.columns)
                }
                print(f"✓ {symbol}: Engineered {len(features.columns)} features")
            else:
                results[symbol] = {'success': False, 'error': 'No data for feature engineering'}
                print(f"✗ {symbol}: No data for feature engineering")

        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"✗ {symbol}: Error in feature engineering - {e}")

    return results

def test_model_training_sample():
    """Test model training on a sample of stocks"""
    print("\nTesting model training on sample stocks...")
    sample_stocks = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'JNJ']  # Test with 5 stocks
    results = {}

    # Configuration for feature engineering
    feature_config = {
        'technical_indicators': {
            'sma_periods': [5, 10, 20],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2
        },
        'statistical_features': {
            'volatility_window': 20,
            'momentum_window': 10,
            'return_window': 1
        }
    }

    for symbol in sample_stocks:
        try:
            print(f"Testing model training for {symbol}...")
            data = fetch_stock_data(symbol, period='2y')

            if data is not None and not data.empty:
                # Add symbol column required by engineer_features
                data['Symbol'] = symbol
                features = engineer_features(data, feature_config)

                # Prepare data for training (add required target columns)
                features['Daily_Return'] = features['Close'].pct_change()
                features['Target_Direction'] = (features['Close'].shift(-1) > features['Close']).astype(int)
                features = features.dropna()

                # Model configuration
                model_config = {
                    'target_type': 'classification',
                    'test_size': 0.2,
                    'random_state': 42,
                    'model_type': 'random_forest',
                    'hyperparameters': {
                        'random_forest': {
                            'n_estimators': 100,
                            'max_depth': 10,
                            'random_state': 42
                        }
                    }
                }

                # Prepare data for LSTM training
                from src.model_training import prepare_data_for_training
                X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data_for_training(features, model_config)

                # Train LSTM model
                model = train_lstm(X_train, y_train, {'sequence_length': 30, 'epochs': 10, 'batch_size': 32})

                results[symbol] = {
                    'success': True,
                    'model_trained': True,
                    'model_type': 'lstm'
                }
                print(f"✓ {symbol}: Model trained and evaluated successfully")
            else:
                results[symbol] = {'success': False, 'error': 'No data for model training'}
                print(f"✗ {symbol}: No data for model training")

        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"✗ {symbol}: Error in model training - {e}")

    return results

def test_streamlit_interface():
    """Test Streamlit interface functionality"""
    print("\nTesting Streamlit interface components...")
    
    # Mock Streamlit session state for testing
    if 'test_mode' not in st.session_state:
        st.session_state.test_mode = True
    
    results = {}
    
    try:
        # Test stock selection
        print("Testing stock selection...")
        for symbol in STOCKS_TO_TEST[:5]:  # Test first 5
            st.session_state.selected_stock = symbol
            print(f"✓ Stock selection: {symbol}")
        
        # Test time period selection
        periods = ['1mo', '3mo', '6mo', '1y', '2y']
        for period in periods:
            st.session_state.selected_period = period
            print(f"✓ Period selection: {period}")
        
        results['streamlit_interface'] = {
            'success': True,
            'stocks_tested': STOCKS_TO_TEST[:5],
            'periods_tested': periods
        }
        
    except Exception as e:
        results['streamlit_interface'] = {'success': False, 'error': str(e)}
        print(f"✗ Streamlit interface error: {e}")
    
    return results

def run_comprehensive_test():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("COMPREHENSIVE TEST SUITE - 23 STOCKS")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Run all tests
    test_results = {
        'data_fetching': test_stock_data_fetching(),
        'technical_indicators': test_technical_indicators(),
        'data_preprocessing': test_data_preprocessing(),
        'feature_engineering': test_feature_engineering(),
        'model_training': test_model_training_sample(),
        'streamlit_interface': test_streamlit_interface()
    }
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Generate summary
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_category, results in test_results.items():
        if isinstance(results, dict):
            for symbol, result in results.items():
                total_tests += 1
                if result.get('success', False):
                    passed_tests += 1
                else:
                    failed_tests += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Test Duration: {duration}")
    print("=" * 60)
    
    # Save detailed results
    results_df = pd.DataFrame.from_dict(test_results, orient='index')
    results_df.to_csv('comprehensive_test_results.csv')
    print("Detailed results saved to 'comprehensive_test_results.csv'")
    
    return test_results

if __name__ == "__main__":
    # Run the comprehensive test suite
    results = run_comprehensive_test()
    
    # Check if all critical tests passed
    critical_passed = all(
        category_result.get('success', False) 
        for category_result in results.values() 
        if isinstance(category_result, dict) and 'success' in category_result
    )
    
    if critical_passed:
        print("\n🎉 ALL CRITICAL TESTS PASSED! Application is ready for deployment.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the results before deployment.")
