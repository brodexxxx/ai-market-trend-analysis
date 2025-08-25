from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from model_training import train_model, load_model

app = Flask(__name__)

# Global variables to store loaded models
loaded_models = {}

def load_all_models():
    """Load all available models from the models directory"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return
    
    # Load traditional ML models
    if os.path.exists('models/trained_model.pkl'):
        try:
            model_data = joblib.load('models/trained_model.pkl')
            loaded_models['traditional'] = model_data
            print("Loaded traditional ML model")
        except Exception as e:
            print(f"Error loading traditional model: {e}")
    
    # Load LSTM model
    if os.path.exists('models/lstm_model.h5'):
        try:
            lstm_model = load_model('models/lstm_model.h5')
            loaded_models['lstm'] = lstm_model
            print("Loaded LSTM model")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")

# Load models on startup
load_all_models()

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        features_data = data['features']
        model_config = data.get('model_config', {})
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(features_data)
        
        # Train the model
        result = train_model(features_df, model_config)
        
        # Reload models after training
        load_all_models()
        
        return jsonify({
            'accuracy': result['accuracy'],
            'model_type': model_config.get('model_type', 'random_forest'),
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': list(loaded_models.keys())})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'traditional')
        
        if model_type not in loaded_models:
            return jsonify({'error': f'Model type {model_type} not loaded'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        if model_type == 'traditional':
            model_data = loaded_models['traditional']
            features_scaled = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(features_scaled)
            return jsonify({'prediction': int(prediction[0]), 'model_type': 'traditional'})
        
        elif model_type == 'lstm':
            # For LSTM, we need to reshape for sequence input
            sequence_length = 30  # Default sequence length
            features_reshaped = features.reshape(1, sequence_length, -1)
            prediction = loaded_models['lstm'].predict(features_reshaped)
            return jsonify({'prediction': int(prediction[0] > 0.5), 'model_type': 'lstm'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
