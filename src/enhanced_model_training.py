import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.feature_selection import (
    RFE, SelectKBest, mutual_info_classif, f_classif,
    SelectFromModel
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform
import joblib
import os
import warnings
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shap
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    """
    Enhanced model training class with ensemble methods and advanced techniques
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_accuracy = 0.0
        self.feature_selector = None
        self.scaler = None
        self.calibrated_models = {}

    def prepare_data_for_training(self, features, target_type='classification', test_size=0.2):
        """
        Enhanced data preparation with advanced feature selection
        """
        # Select features and target
        feature_columns = [col for col in features.columns if col not in [
            'Target_Direction', 'Target_Class', 'Target_Return', 'Symbol', 'Date'
        ] and not col.startswith('target_') and features[col].dtype != 'datetime64[ns]']

        X = features[feature_columns]

        # Select target based on configuration
        if target_type == 'classification':
            y = features['Target_Direction']
        elif target_type == 'multiclass':
            le = LabelEncoder()
            y = le.fit_transform(features['Target_Class'])
        else:  # regression
            y = features['Target_Return']

        # Handle duplicate indices
        X = X[~X.index.duplicated(keep='first')]
        y = y.loc[X.index]

        # Handle missing values
        X = X.dropna()
        y = y.loc[X.index]

        # Time series split for better validation
        tscv = TimeSeriesSplit(n_splits=5)
        split_index = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

    def advanced_feature_selection(self, X, y, method='mutual_info', k=20):
        """
        Advanced feature selection methods
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k)
        elif method == 'from_model':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=estimator, max_features=k)

        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)

        self.feature_selector = selector
        return X_selected, selected_features

    def create_base_models(self):
        """
        Create base models for ensemble including advanced ML algorithms
        """
        models = {
            'logistic_regression': LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100, depth=6, learning_rate=0.1,
                random_state=42, verbose=False
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', alpha=0.0001, max_iter=500,
                random_state=42
            ),
            'svm': SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True,
                random_state=42
            ),
            'naive_bayes': GaussianNB()
        }

        return models

    def create_ensemble_models(self, base_models):
        """
        Create ensemble models using voting and stacking
        """
        ensemble_models = {}

        # Voting Classifier (Hard Voting)
        voting_hard = VotingClassifier(
            estimators=list(base_models.items()),
            voting='hard'
        )
        ensemble_models['voting_hard'] = voting_hard

        # Voting Classifier (Soft Voting)
        voting_soft = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()
                       if hasattr(model, 'predict_proba')],
            voting='soft'
        )
        ensemble_models['voting_soft'] = voting_soft

        # Stacking Classifier
        stacking = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        ensemble_models['stacking'] = stacking

        return ensemble_models

    def hyperparameter_tuning(self, X_train, y_train, model_name, model):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 40]
            },
            'catboost': {
                'iterations': [50, 100, 200],
                'depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }

        if model_name in param_grids:
            param_grid = param_grids[model_name]

            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model, param_grid, n_iter=10, cv=3,
                scoring='accuracy', random_state=42, n_jobs=-1
            )

            search.fit(X_train, y_train)
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best cross-validation score: {search.best_score_:.4f}")

            return search.best_estimator_
        else:
            return model

    def calibrate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Calibrate model for better probability estimates
        """
        try:
            calibrated_model = CalibratedClassifierCV(
                model, method='isotonic', cv=3
            )
            calibrated_model.fit(X_train, y_train)

            # Test calibration
            y_prob = calibrated_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)

            print(f"Model calibrated. ROC-AUC: {auc_score:.4f}")
            return calibrated_model
        except Exception as e:
            print(f"Calibration failed: {e}")
            return model

    def train_ensemble_system(self, X_train, y_train, X_test, y_test, feature_columns):
        """
        Train the complete ensemble system
        """
        print("🚀 Starting Enhanced Model Training System")
        print("=" * 50)

        # Step 1: Advanced Feature Selection
        print("📊 Performing Advanced Feature Selection...")
        X_train_selected, selected_indices = self.advanced_feature_selection(
            X_train, y_train, method='mutual_info', k=20
        )
        X_test_selected = self.feature_selector.transform(X_test)

        selected_features = [feature_columns[i] for i in selected_indices]
        print(f"Selected {len(selected_features)} features: {selected_features}")

        # Step 2: Create and train base models
        print("🏗️ Creating Base Models...")
        base_models = self.create_base_models()

        model_results = {}
        for name, model in base_models.items():
            print(f"Training {name}...")
            try:
                # Hyperparameter tuning for other models
                tuned_model = self.hyperparameter_tuning(X_train_selected, y_train, name, model)
                tuned_model.fit(X_train_selected, y_train)
                y_pred = tuned_model.predict(X_test_selected)
                y_test_lstm = y_test

                accuracy = accuracy_score(y_test_lstm, y_pred)
                model_results[name] = {
                    'model': tuned_model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }

                print(f"✅ {name} accuracy: {accuracy:.4f}")

                # Calibrate model if it supports predict_proba
                if hasattr(tuned_model, 'predict_proba'):
                    calibrated_model = self.calibrate_model(tuned_model, X_train_selected, y_train,
                                                          X_test_selected, y_test_lstm)
                    self.calibrated_models[name] = calibrated_model

            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

        # Step 3: Create and train ensemble models
        print("🎯 Creating Ensemble Models...")
        ensemble_models = self.create_ensemble_models(base_models)

        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                accuracy = accuracy_score(y_test, y_pred)

                model_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }

                print(f"✅ {name} accuracy: {accuracy:.4f}")

            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

        # Step 4: Select best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        self.best_model = model_results[best_model_name]['model']
        self.best_accuracy = model_results[best_model_name]['accuracy']

        print("\n🏆 Best Model Results:")
        print(f"Model: {best_model_name}")
        print(f"Accuracy: {self.best_accuracy:.4f}")
        print(f"Improvement: {((self.best_accuracy - 0.5) * 100):.1f}% over random")

        # Step 5: Save enhanced model system
        self.save_enhanced_model(model_results, selected_features)

        return {
            'model_results': model_results,
            'best_model': self.best_model,
            'best_accuracy': self.best_accuracy,
            'selected_features': selected_features,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }

    def save_enhanced_model(self, model_results, selected_features):
        """
        Save the enhanced model system
        """
        os.makedirs('models', exist_ok=True)

        enhanced_model_data = {
            'model_results': model_results,
            'best_model': self.best_model,
            'best_accuracy': self.best_accuracy,
            'selected_features': selected_features,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'calibrated_models': self.calibrated_models
        }

        joblib.dump(enhanced_model_data, 'models/enhanced_model_system.pkl')
        print("💾 Enhanced model system saved successfully!")

    def load_enhanced_model(self):
        """
        Load the enhanced model system
        """
        try:
            model_data = joblib.load('models/enhanced_model_system.pkl')
            self.best_model = model_data['best_model']
            self.best_accuracy = model_data['best_accuracy']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.calibrated_models = model_data.get('calibrated_models', {})
            return model_data
        except FileNotFoundError:
            print("❌ Enhanced model system not found!")
            return None

    def predict_with_best_model(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Please train or load a model first.")

        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        # Make prediction
        if hasattr(self.best_model, 'predict_proba'):
            y_prob = self.best_model.predict_proba(X_selected)
            y_pred = self.best_model.predict(X_selected)
            return y_pred, y_prob
        else:
            y_pred = self.best_model.predict(X_selected)
            return y_pred, None

    def explain_model_predictions(self, X, feature_names=None, max_evals=1000):
        """
        Explain model predictions using SHAP values
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Please train or load a model first.")

        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        # Get feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]

        try:
            # Create SHAP explainer based on model type
            if hasattr(self.best_model, 'predict_proba'):
                if hasattr(self.best_model, 'feature_importances_'):
                    # Tree-based models
                    explainer = shap.TreeExplainer(self.best_model)
                    shap_values = explainer.shap_values(X_selected)
                else:
                    # Other models
                    explainer = shap.Explainer(self.best_model, X_selected[:100])  # Use subset for background
                    shap_values = explainer(X_selected)

                # Calculate feature importance
                if isinstance(shap_values, list):
                    # Multi-class case
                    feature_importance = np.abs(shap_values[0]).mean(axis=0)
                else:
                    # Binary classification or regression
                    feature_importance = np.abs(shap_values).mean(axis=0)

                # Create explanation summary
                explanation = {
                    'shap_values': shap_values,
                    'feature_importance': feature_importance,
                    'feature_names': feature_names,
                    'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
                }

                return explanation
            else:
                print("Model does not support probability predictions, SHAP explanation not available")
                return None

        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None

    def get_model_performance_metrics(self, X_test, y_test, model_results):
        """
        Calculate comprehensive performance metrics for all models
        """
        metrics = {}

        for model_name, model_data in model_results.items():
            model = model_data['model']
            y_pred = model_data['predictions']

            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            # ROC-AUC if applicable
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_prob)

                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    pr_auc = auc(recall, precision)
                except:
                    roc_auc = None
                    pr_auc = None
            else:
                roc_auc = None
                pr_auc = None

            metrics[model_name] = {
                'accuracy': accuracy,
                'precision': class_report['weighted avg']['precision'],
                'recall': class_report['weighted avg']['recall'],
                'f1_score': class_report['weighted avg']['f1-score'],
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report
            }

        return metrics

    def detect_overfitting(self, X_train, y_train, X_test, y_test):
        """
        Detect potential overfitting in the model
        """
        if self.best_model is None:
            return None

        # Train accuracy
        y_train_pred = self.best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Test accuracy
        y_test_pred = self.best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Overfitting indicators
        overfitting_ratio = train_accuracy / test_accuracy if test_accuracy > 0 else float('inf')
        overfitting_gap = train_accuracy - test_accuracy

        overfitting_analysis = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_gap': overfitting_gap,
            'is_overfitting': overfitting_gap > 0.1,  # More than 10% gap
            'severity': 'high' if overfitting_gap > 0.2 else 'medium' if overfitting_gap > 0.1 else 'low'
        }

        return overfitting_analysis

def train_enhanced_model(features, config=None):
    """
    Main function to train enhanced model system
    """
    if config is None:
        config = {
            'target_type': 'classification',
            'test_size': 0.2,
            'feature_selection': 'mutual_info',
            'k_features': 20
        }

    trainer = EnhancedModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test, feature_columns = trainer.prepare_data_for_training(
        features, config.get('target_type', 'classification'), config.get('test_size', 0.2)
    )

    # Train enhanced system
    results = trainer.train_ensemble_system(X_train, y_train, X_test, y_test, feature_columns)

    return results

if __name__ == "__main__":
    # Test the enhanced model training
    import yfinance as yf

    print("🧪 Testing Enhanced Model Training System")

    # Fetch sample data
    df = yf.download('AAPL', period='2y', interval='1d')
    df['Symbol'] = 'AAPL'
    df['Daily_Return'] = df['Close'].pct_change()

    # Create basic features
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                   df['Close'].diff().clip(upper=0).abs().rolling(window=14).mean())))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['BB_Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    # Train enhanced model
    results = train_enhanced_model(df)

    print("\n🎉 Enhanced Model Training Complete!")
    print(f"Best Model Accuracy: {results['best_accuracy']:.4f}")
