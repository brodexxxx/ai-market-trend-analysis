import pandas as pd
import numpy as np
import yaml
import json
import pickle
import joblib
from datetime import datetime, timedelta
import os
import logging

def setup_logging(log_file='market_analysis.log'):
    """
    Set up logging configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def save_config(config, config_path='config.yaml'):
    """
    Save configuration to YAML file
    """
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving config file: {e}")
        return False

def create_directory_structure():
    """
    Create necessary directories for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'reports',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def save_dataframe(df, file_path, format='csv'):
    """
    Save DataFrame to file in specified format
    """
    try:
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format == 'pickle':
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_dataframe(file_path, format='csv'):
    """
    Load DataFrame from file
    """
    try:
        if format == 'csv':
            df = pd.read_csv(file_path)
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
        elif format == 'pickle':
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_performance_metrics(true_values, predictions, model_type='classification'):
    """
    Calculate performance metrics based on model type
    """
    if model_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': accuracy_score(true_values, predictions),
            'precision': precision_score(true_values, predictions, average='weighted'),
            'recall': recall_score(true_values, predictions, average='weighted'),
            'f1_score': f1_score(true_values, predictions, average='weighted')
        }
    elif model_type == 'regression':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            'mse': mean_squared_error(true_values, predictions),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions)
        }

def format_currency(value):
    """
    Format number as currency
    """
    return f"${value:,.2f}"

def format_percentage(value):
    """
    Format number as percentage
    """
    return f"{value:.2%}"

def get_date_ranges(period='1y'):
    """
    Get date ranges based on period string
    """
    end_date = datetime.now()
    
    if period == '1d':
        start_date = end_date - timedelta(days=1)
    elif period == '1w':
        start_date = end_date - timedelta(weeks=1)
    elif period == '1m':
        start_date = end_date - timedelta(days=30)
    elif period == '3m':
        start_date = end_date - timedelta(days=90)
    elif period == '6m':
        start_date = end_date - timedelta(days=180)
    elif period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == '2y':
        start_date = end_date - timedelta(days=730)
    elif period == '5y':
        start_date = end_date - timedelta(days=1825)
    else:
        start_date = end_date - timedelta(days=365)  # Default to 1 year
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for excessive missing values
    missing_percentage = df.isnull().mean()
    high_missing = missing_percentage[missing_percentage > 0.3]
    if not high_missing.empty:
        return False, f"High missing values in columns: {high_missing.index.tolist()}"
    
    return True, "DataFrame validation passed"

def create_summary_statistics(df):
    """
    Create summary statistics for DataFrame
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary

def export_results(results, file_path, format='json'):
    """
    Export results to file
    """
    try:
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'csv':
            if isinstance(results, pd.DataFrame):
                results.to_csv(file_path, index=False)
            else:
                raise ValueError("Results must be a DataFrame for CSV export")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {file_path}")
        return True
    except Exception as e:
        print(f"Error exporting results: {e}")
        return False

if __name__ == "__main__":
    # Test utility functions
    logger = setup_logging()
    logger.info("Testing utility functions")
    
    # Test directory creation
    create_directory_structure()
    
    # Test date ranges
    start, end = get_date_ranges('1y')
    print(f"Date range: {start} to {end}")
    
    # Test summary statistics
    sample_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    summary = create_summary_statistics(sample_df)
    print("Summary statistics:")
    print(json.dumps(summary, indent=2))
