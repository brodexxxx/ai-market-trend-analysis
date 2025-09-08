# Quick Start Instructions

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Run the Complete Pipeline (Recommended)
```bash
python main.py
```
This will:
- Fetch stock data from Yahoo Finance
- Preprocess and clean the data
- Calculate technical indicators and features
- Train a machine learning model
- Generate evaluation reports

## Step 3: Start the API Service (Optional)
```bash
python src/api_service.py
```
Then test the API:
```bash
python test_api_updated.py
```

## Step 4: View Results
Check the generated files:
- `reports/` - Performance reports and visualizations
- `models/trained_model.pkl` - Trained model file
- `data/` - Processed data and features

## Quick Commands Summary

1. **Full pipeline**: `python main.py`
2. **API service**: `python src/api_service.py`
3. **Test API**: `python test_api_updated.py`
4. **Test LSTM**: `python test_lstm_model.py`

The model has already been trained with 55.15% accuracy and is ready to use!
