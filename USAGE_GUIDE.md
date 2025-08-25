# AI Market Trend Analysis - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python main.py
```

### 3. Start the API Service
```bash
python src/api_service.py
```

## Detailed Usage

### Data Collection and Preprocessing
```python
from src.data_preprocessing import preprocess_data

# Configure data sources
data_config = {
    'yahoo_finance': {
        'enabled': True,
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'period': '1y',
        'interval': '1d'
    }
}

# Fetch and preprocess data
data = preprocess_data(data_config)
```

### Feature Engineering
```python
from src.feature_engineering import engineer_features

# Configure feature engineering
feature_config = {
    'technical_indicators': {
        'sma_periods': [5, 10, 20, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26
    }
}

# Engineer features
features = engineer_features(data, feature_config)
```

### Model Training
```python
from src.model_training import train_model

# Configure model training
model_config = {
    'model_type': 'random_forest',
    'test_size': 0.2,
    'random_state': 42,
    'hyperparameters': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
}

# Train model
result = train_model(features, model_config)
```

### API Usage

#### Start API Server
```bash
python src/api_service.py
```

#### API Endpoints

**Health Check**
```bash
curl http://127.0.0.1:5000/health
```

**Train Model**
```bash
curl -X POST http://27.0.0.1:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"feature1": 0.1, "feature2": 0.2, "Target_Direction": 1},
      {"feature1": 0.3, "feature2": 0.4, "Target_Direction": 0}
    ],
    "model_config": {
      "model_type": "random_forest",
      "hyperparameters": {
        "random_forest": {"n_estimators": 10, "max_depth": 3}
      }
    }
  }'
```

**Make Prediction**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2],
    "model_type": "traditional"
  }'
```

## Configuration

Edit `config.yaml` to customize:
- Data sources (Yahoo Finance, Alpha Vantage)
- Feature engineering parameters
- Model hyperparameters
- Evaluation metrics

## Output Files

- **Processed Data**: `data/processed/processed_stock_data.csv`
- **Engineered Features**: `data/features/engineered_features.csv`
- **Trained Models**: `models/trained_model.pkl`
- **Evaluation Reports**: `reports/` directory

## Testing

Run comprehensive tests:
```bash
python test_api_updated.py
python test_lstm_model.py
```

## Supported Models

1. **Traditional ML Models**:
   - Logistic Regression
   - Random Forest
   - XGBoost

2. **Deep Learning Models**:
   - LSTM (Long Short-Term Memory)
   - 1D CNN (Convolutional Neural Network)

## Data Sources

- Yahoo Finance API (real-time stock data)
- Alpha Vantage API (alternative data source)
- Custom CSV files (historical data)

## Features Calculated

- Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Statistical Features (volatility, momentum, returns)
- Lag Features for time series analysis
- Custom feature engineering
