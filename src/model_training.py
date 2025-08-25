import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # pyright: ignore[reportMissingModuleSource]
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource]
from xgboost import XGBClassifier # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler, LabelEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import accuracy_score, classification_report # pyright: ignore[reportMissingModuleSource]
import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
import joblib
import os

def prepare_data_for_training(features, model_config):
    """
    Prepare data for model training
    """
    # Select features and target
    feature_columns = [col for col in features.columns if col not in [
        'Target_Direction', 'Target_Class', 'Target_Return', 'Symbol', 'Date'
    ] and not col.startswith('target_')]
    
    X = features[feature_columns]
    
    # Select target based on configuration
    target_type = model_config.get('target_type', 'classification')
    
    if target_type == 'classification':
        y = features['Target_Direction']
    elif target_type == 'multiclass':
        le = LabelEncoder()
        y = le.fit_transform(features['Target_Class'])
    else:  # regression
        y = features['Target_Return']
    
    # Handle duplicate indices
    X = X[~X.index.duplicated(keep='first')]
    y = y.loc[X.index]  # Use .loc for proper indexing
    
    # Handle missing values
    X = X.dropna()
    y = y.loc[X.index]  # Use .loc for proper indexing
    
    # Split data
    test_size = model_config.get('test_size', 0.2)
    random_state = model_config.get('random_state', 42)
    
    if model_config.get('time_series_split', False):
        # Time series split (for time series data)
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if target_type != 'regression' else None
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def train_logistic_regression(X_train, y_train, hyperparams):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(
        C=hyperparams.get('C', 1.0),
        penalty=hyperparams.get('penalty', 'l2'),
        solver=hyperparams.get('solver', 'liblinear'),
        random_state=hyperparams.get('random_state', 42)
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, hyperparams):
    """
    Train Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=hyperparams.get('n_estimators', 100),
        max_depth=hyperparams.get('max_depth', None),
        min_samples_split=hyperparams.get('min_samples_split', 2),
        min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
        random_state=hyperparams.get('random_state', 42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, hyperparams):
    """
    Train XGBoost model
    """
    model = XGBClassifier(
        n_estimators=hyperparams.get('n_estimators', 100),
        max_depth=hyperparams.get('max_depth', 6),
        learning_rate=hyperparams.get('learning_rate', 0.1),
        subsample=hyperparams.get('subsample', 0.8),
        random_state=hyperparams.get('random_state', 42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    model.save(filename)

def load_model(filename):
    """
    Load a trained model from a file.
    """
    return tf.keras.models.load_model(filename)

def train_lstm(X_train, y_train, hyperparams):
    """
    Train LSTM model for time series
    """
    # Reshape data for LSTM [samples, timesteps, features]
    sequence_length = hyperparams.get('sequence_length', 30)
    X_reshaped = reshape_for_lstm(X_train, sequence_length)
    
    model = Sequential([
        LSTM(hyperparams.get('units', 50), 
             input_shape=(sequence_length, X_reshaped.shape[2]),
             dropout=hyperparams.get('dropout', 0.2),
             recurrent_dropout=hyperparams.get('recurrent_dropout', 0.2)),
        Dense(25, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=hyperparams.get('learning_rate', 0.001)),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        X_reshaped, y_train[sequence_length:],
        epochs=hyperparams.get('epochs', 50),
        batch_size=hyperparams.get('batch_size', 32),
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model after training
    save_model(model, 'models/lstm_model.h5')
    
    return model

def reshape_for_lstm(X, sequence_length):
    """
    Reshape data for LSTM input
    """
    X_reshaped = []
    for i in range(sequence_length, len(X)):
        X_reshaped.append(X[i-sequence_length:i])
    return np.array(X_reshaped)

def train_model(features, model_config):
    """
    Main model training function
    """
    print("Starting model training...")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data_for_training(features, model_config)
    
    # Get model type from configuration
    model_type = model_config.get('model_type', 'random_forest')
    hyperparams = model_config.get('hyperparameters', {}).get(model_type, {})
    
    # Train selected model
    if model_type == 'logistic_regression':
        model = train_logistic_regression(X_train, y_train, hyperparams)
    elif model_type == 'random_forest':
        model = train_random_forest(X_train, y_train, hyperparams)
    elif model_type == 'xgboost':
        model = train_xgboost(X_train, y_train, hyperparams)
    elif model_type == 'lstm':
        model = train_lstm(X_train, y_train, hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions
    if model_type == 'lstm':
        # For LSTM, we need to prepare test data differently
        sequence_length = hyperparams.get('sequence_length', 30)
        X_test_reshaped = reshape_for_lstm(X_test, sequence_length)
        y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
        y_test = y_test[sequence_length:]
    else:
        y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test accuracy: {accuracy:.4f}")
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'model_type': model_type
    }, 'models/trained_model.pkl')
    
    print("Model saved successfully.")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy
    }

def cross_validate_model(features, model_config, cv_folds=5):
    """
    Perform cross-validation
    """
    X_train, _, y_train, _, _, _ = prepare_data_for_training(features, model_config)
    
    model_type = model_config.get('model_type', 'random_forest')
    hyperparams = model_config.get('hyperparameters', {}).get(model_type, {})
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(**hyperparams)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**hyperparams)
    elif model_type == 'xgboost':
        model = XGBClassifier(**hyperparams)
    else:
        raise ValueError(f"Cross-validation not implemented for {model_type}")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

if __name__ == "__main__":
    # Test the model training
    import yfinance as yf
    from src.feature_engineering import engineer_features
    
    # Fetch sample data
    df = yf.download('AAPL', period='1y', interval='1d')
    df['Symbol'] = 'AAPL'
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    # Test configuration
    config = {
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
    
    result = train_model(df, config)
    print(f"Final accuracy: {result['accuracy']:.4f}")
