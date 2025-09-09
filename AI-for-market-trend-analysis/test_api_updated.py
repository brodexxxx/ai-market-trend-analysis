import requests
import json
import numpy as np
import pytest

# Test the /health endpoint
def test_health():
    url = 'http://127.0.0.1:5000/health'
    try:
        response = requests.get(url, timeout=5)
        assert response.status_code == 200
        print('Health Response:', response.json())
    except requests.exceptions.ConnectionError:
        pytest.skip("API server is not running - skipping health check test")
    except requests.exceptions.Timeout:
        pytest.skip("API server timeout - skipping health check test")

# Test the /train endpoint
def test_train():
    url = 'http://127.0.0.1:5000/train'
    try:
        # Sample feature data for training - using traditional model instead of LSTM
        features = [
            {'feature1': 0.1, 'feature2': 0.2, 'Target_Direction': 1},
            {'feature1': 0.3, 'feature2': 0.4, 'Target_Direction': 0},
            {'feature1': 0.5, 'feature2': 0.6, 'Target_Direction': 1},
            {'feature1': 0.7, 'feature2': 0.8, 'Target_Direction': 0},
            {'feature1': 0.9, 'feature2': 1.0, 'Target_Direction': 1},
            {'feature1': 1.1, 'feature2': 1.2, 'Target_Direction': 0},
            {'feature1': 1.3, 'feature2': 1.4, 'Target_Direction': 1},
            {'feature1': 1.5, 'feature2': 1.6, 'Target_Direction': 0}
        ]
        model_config = {
            'model_type': 'random_forest',
            'hyperparameters': {
                'random_forest': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            }
        }
        response = requests.post(url, json={'features': features, 'model_config': model_config}, timeout=10)
        assert response.status_code == 200
        print('Train Response:', response.json())
    except requests.exceptions.ConnectionError:
        pytest.skip("API server is not running - skipping train test")
    except requests.exceptions.Timeout:
        pytest.skip("API server timeout - skipping train test")

# Test the /predict endpoint
def test_predict():
    url = 'http://127.0.0.1:5000/predict'
    try:
        # Sample feature data for prediction - use traditional model since it's loaded
        features = [0.1, 0.2]  # Replace with actual feature values
        response = requests.post(url, json={'features': features, 'model_type': 'traditional'}, timeout=10)
        assert response.status_code == 200
        print('Predict Response:', response.json())
    except requests.exceptions.ConnectionError:
        pytest.skip("API server is not running - skipping predict test")
    except requests.exceptions.Timeout:
        pytest.skip("API server timeout - skipping predict test")

if __name__ == '__main__':
    test_health()
    test_train()
    test_predict()
