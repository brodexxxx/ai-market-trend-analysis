import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Evaluate classification model performance
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    
    # ROC AUC if probabilities are available
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics

def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression model performance
    """
    metrics = {}
    
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or ['Down', 'Up'],
                yticklabels=class_names or ['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importance attribute")
        return
    
    # Get top N features
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importance[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.close()

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png')
    plt.close()
    
    return roc_auc

def create_interactive_plot(predictions_df):
    """
    Create interactive plot with Plotly
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Price and Predictions', 'Returns'))
    
    # Price and predictions
    fig.add_trace(
        go.Scatter(x=predictions_df.index, y=predictions_df['Close'], name='Close Price'),
        row=1, col=1
    )
    
    # Add buy/sell signals
    buy_signals = predictions_df[predictions_df['Prediction'] == 1]
    sell_signals = predictions_df[predictions_df['Prediction'] == 0]
    
    fig.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                  mode='markers', name='Buy Signal', marker=dict(color='green', size=8)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                  mode='markers', name='Sell Signal', marker=dict(color='red', size=8)),
        row=1, col=1
    )
    
    # Returns
    fig.add_trace(
        go.Scatter(x=predictions_df.index, y=predictions_df['Actual_Return'], name='Actual Return'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=predictions_df.index, y=predictions_df['Predicted_Return'], name='Predicted Return'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Market Trend Analysis")
    fig.write_html('reports/interactive_analysis.html')

def generate_performance_report(metrics, model_type='classification'):
    """
    Generate comprehensive performance report
    """
    report = "# Model Performance Report\n\n"
    
    if model_type == 'classification':
        report += "## Classification Metrics\n\n"
        report += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
        report += f"- **Precision**: {metrics['precision']:.4f}\n"
        report += f"- **Recall**: {metrics['recall']:.4f}\n"
        report += f"- **F1-Score**: {metrics['f1_score']:.4f}\n"
        
        if 'roc_auc' in metrics:
            report += f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n"
        
        report += "\n## Classification Report\n\n"
        report += "```\n"
        report += classification_report(metrics['y_true'], metrics['y_pred'])
        report += "```\n"
        
    else:  # regression
        report += "## Regression Metrics\n\n"
        report += f"- **MSE**: {metrics['mse']:.6f}\n"
        report += f"- **RMSE**: {metrics['rmse']:.6f}\n"
        report += f"- **MAE**: {metrics['mae']:.6f}\n"
        report += f"- **R²**: {metrics['r2']:.4f}\n"
    
    # Add confusion matrix if available
    if 'confusion_matrix' in metrics:
        report += "\n## Confusion Matrix\n\n"
        report += f"```\n{metrics['confusion_matrix']}\n```\n"
    
    return report

def evaluate_model(training_result, features):
    """
    Main evaluation function
    """
    print("Starting model evaluation...")
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Extract results
    model = training_result['model']
    X_test = training_result['X_test']
    y_test = training_result['y_test']
    y_pred = training_result['y_pred']
    
    # Determine model type
    model_type = 'classification'  # Default, can be extended for regression
    
    # Evaluate model
    if model_type == 'classification':
        metrics = evaluate_classification_model(y_test, y_pred)
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])
        
        # Plot feature importance if available
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            plot_feature_importance(model, training_result['feature_columns'])
        
        # Plot ROC curve if probabilities are available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = plot_roc_curve(y_test, y_prob)
            metrics['roc_auc'] = roc_auc
    
    # Generate performance report
    metrics['y_true'] = y_test
    metrics['y_pred'] = y_pred
    report = generate_performance_report(metrics, model_type)
    
    # Save report
    with open('reports/performance_report.md', 'w') as f:
        f.write(report)
    
    # Create predictions DataFrame for visualization
    predictions_df = pd.DataFrame({
        'Close': features.loc[y_test.index, 'Close'],
        'Prediction': y_pred,
        'Actual_Direction': y_test,
        'Actual_Return': features.loc[y_test.index, 'Daily_Return'],
        'Predicted_Return': np.where(y_pred == 1, 
                                   features.loc[y_test.index, 'Daily_Return'].mean() * 1.1, 
                                   features.loc[y_test.index, 'Daily_Return'].mean() * 0.9)
    })
    
    # Create interactive plot
    create_interactive_plot(predictions_df)
    
    print("Model evaluation completed. Reports saved in 'reports/' directory.")
    
    return metrics

def backtest_strategy(predictions_df, initial_capital=10000):
    """
    Backtest trading strategy based on predictions
    """
    capital = initial_capital
    position = 0
    trades = []
    
    for i, (idx, row) in enumerate(predictions_df.iterrows()):
        if row['Prediction'] == 1 and position == 0:  # Buy signal
            position = capital / row['Close']
            capital = 0
            trades.append({'date': idx, 'action': 'BUY', 'price': row['Close']})
        elif row['Prediction'] == 0 and position > 0:  # Sell signal
            capital = position * row['Close']
            position = 0
            trades.append({'date': idx, 'action': 'SELL', 'price': row['Close']})
    
    # Final valuation
    if position > 0:
        final_value = position * predictions_df.iloc[-1]['Close']
    else:
        final_value = capital
    
    return {
        'final_value': final_value,
        'return_pct': (final_value - initial_capital) / initial_capital * 100,
        'trades': trades,
        'sharpe_ratio': calculate_sharpe_ratio(predictions_df['Actual_Return'])
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

if __name__ == "__main__":
    # Test the evaluation
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_classification_model(y_test, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
