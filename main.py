import os
import yaml
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.evaluation import evaluate_model

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    data = preprocess_data(config['data_sources'])

    # Feature engineering
    features = engineer_features(data, config['features'])

    # Train model
    model = train_model(features, config['models'])

    # Evaluate model
    evaluation_results = evaluate_model(model, features)

    # Save results
    if config['output']['save_models']:
        joblib.dump(model['model'], os.path.join(config['output']['models_dir'], 'trained_model.pkl'))

    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()
