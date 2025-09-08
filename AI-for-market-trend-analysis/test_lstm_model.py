import yaml
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print('=== LSTM MODEL TEST ===')

# Load and preprocess data
data = preprocess_data(config['data_sources'])

# Feature engineering
features = engineer_features(data, config['features'])

# Update model configuration for LSTM
lstm_config = config.copy()
lstm_config['models']['model_type'] = 'lstm'

# Train LSTM model
try:
    lstm_result = train_model(features, lstm_config['models'])
    print(f'LSTM Model Accuracy: {lstm_result["accuracy"]:.4f}')
except Exception as e:
    print(f'LSTM Model Error: {e}')
