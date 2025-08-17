import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Tuple, Optional, Dict, Any, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Advanced feature engineering for RUL prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw sensor data"""
        df_features = df.copy()
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        setting_cols = [col for col in df.columns if col.startswith('setting_')]
        
        if 'cycle' not in df.columns:
            logger.warning("No 'cycle' column found, creating sequential cycles")
            df_features['cycle'] = range(1, len(df) + 1)
        
        cycle_col = df_features['cycle']
        
        # Cycle-based features
        df_features['cycle_norm'] = cycle_col / cycle_col.max()
        df_features['cycle_squared'] = cycle_col ** 2
        df_features['cycle_sqrt'] = np.sqrt(cycle_col)
        df_features['cycle_log'] = np.log1p(cycle_col)
        df_features['cycle_inv'] = 1.0 / (cycle_col + 1)
        
        # Rolling statistics for sensor data (window of 5 cycles)
        window_size = min(5, len(df_features))
        
        for col in sensor_cols:
            if col in df_features.columns:
                # Rolling mean and std
                df_features[f'{col}_rolling_mean'] = df_features[col].rolling(window=window_size, min_periods=1).mean()
                df_features[f'{col}_rolling_std'] = df_features[col].rolling(window=window_size, min_periods=1).std().fillna(0)
                
                # Trend features (difference from rolling mean)
                df_features[f'{col}_trend'] = df_features[col] - df_features[f'{col}_rolling_mean']
                
                # Rate of change
                df_features[f'{col}_rate'] = df_features[col].diff().fillna(0)
        
        # Engine degradation features
        if 'engine_id' in df_features.columns:
            # Max cycle per engine (proxy for engine life)
            engine_max_cycles = df_features.groupby('engine_id')['cycle'].transform('max')
            df_features['remaining_cycles'] = engine_max_cycles - cycle_col
            df_features['cycle_ratio'] = cycle_col / engine_max_cycles
        
        # Sensor interaction features (top combinations)
        important_sensors = sensor_cols[:10]  # Use first 10 sensors to avoid explosion
        for i, sensor1 in enumerate(important_sensors):
            if sensor1 in df_features.columns:
                for sensor2 in important_sensors[i+1:i+3]:  # Limit combinations
                    if sensor2 in df_features.columns:
                        df_features[f'{sensor1}_{sensor2}_ratio'] = (
                            df_features[sensor1] / (df_features[sensor2] + 1e-8)
                        )
        
        # Operating condition features
        for col in setting_cols:
            if col in df_features.columns:
                df_features[f'{col}_squared'] = df_features[col] ** 2
        
        # Remove any infinite or extremely large values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        # Store feature names (excluding metadata columns)
        metadata_cols = ['engine_id', 'cycle', 'RUL']
        self.feature_names = [col for col in df_features.columns if col not in metadata_cols]
        
        logger.info(f"Created {len(self.feature_names)} engineered features")
        
        return df_features

class ImprovedRULPredictor:
    """Improved RUL predictor using ensemble methods and advanced feature engineering"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.pipeline = None
        self.feature_engineer = FeatureEngineering()
        self.feature_columns = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.cv_scores = None
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'param_grid': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [10, 15, 20],
                    'model__min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                ),
                'param_grid': {
                    'model__n_estimators': [100, 200],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [6, 8, 10]
                }
            }
        }
    
    def load_training_data(self, filepath: str = 'data/processed/train_processed.csv') -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data with feature engineering"""
        try:
            logger.info(f"Loading training data from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Training data file not found: {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            if df.empty:
                raise ValueError("Training data is empty")
            
            # Validate required columns
            required_cols = ['RUL']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Feature engineering
            df_engineered = self.feature_engineer.create_features(df)
            
            # Extract features and target
            self.feature_columns = self.feature_engineer.feature_names
            
            if not self.feature_columns:
                raise ValueError("No feature columns found after engineering")
            
            X = df_engineered[self.feature_columns].values
            y = df_engineered['RUL'].values
            
            # Validate data
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                logger.warning("Found NaN values in data, cleaning...")
                
                # Remove samples with NaN targets
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                
                # Impute missing features
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Remove negative RUL values (data quality issue)
            valid_rul_mask = y >= 0
            X = X[valid_rul_mask]
            y = y[valid_rul_mask]
            
            logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
            logger.info(f"Target statistics - Mean: {np.mean(y):.1f}, Std: {np.std(y):.1f}, Range: {np.min(y):.1f}-{np.max(y):.1f}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def train_model(self, data_path: str = 'data/processed/train_processed.csv', 
                   tune_hyperparameters: bool = True, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the improved RUL prediction model"""
        
        # Load training data
        X, y = self.load_training_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Get model configuration
        if self.model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        config = self.model_configs[self.model_type]
        base_model = config['model']
        
        # Create pipeline with preprocessing
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('feature_selection', SelectKBest(f_regression, k=min(50, X.shape[1]))),
            ('model', base_model)
        ])
        
        # Hyperparameter tuning
        if tune_hyperparameters and len(X_train) > 100:
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = config['param_grid']
            # Add feature selection parameter
            param_grid['feature_selection__k'] = [min(30, X.shape[1]), min(50, X.shape[1])]
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=min(5, len(X_train) // 50),  # Adaptive CV folds
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {-grid_search.best_score_:.3f}")
        else:
            logger.info("Training with default parameters...")
            self.pipeline.fit(X_train, y_train)
        
        # Cross-validation scores
        self.cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, 
            cv=min(5, len(X_train) // 50),
            scoring='neg_mean_squared_error'
        )
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_rmse_mean': np.sqrt(-self.cv_scores.mean()),
            'cv_rmse_std': np.sqrt(self.cv_scores.std())
        }
        
        logger.info(f"Model training completed:")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.3f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.3f}")
        logger.info(f"Test R²: {test_metrics['r2']:.3f}")
        logger.info(f"CV RMSE: {test_metrics['cv_rmse_mean']:.3f} ± {test_metrics['cv_rmse_std']:.3f}")
        
        return test_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL for input samples"""
        if self.pipeline is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.pipeline.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_single(self, features: List[float]) -> float:
        """Predict RUL for a single sample"""
        X = np.array(features).reshape(1, -1)
        prediction = self.predict(X)[0]
        return float(prediction)
    
    def prepare_features_for_prediction(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Prepare sensor data for prediction"""
        try:
            # Create a DataFrame from sensor data
            df = pd.DataFrame([sensor_data])
            
            # Feature engineering
            df_engineered = self.feature_engineer.create_features(df)
            
            # Extract features in the same order as training
            if self.feature_columns is None:
                raise ValueError("Model not trained yet - no feature columns defined")
            
            # Handle missing features
            for col in self.feature_columns:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0
            
            X = df_engineered[self.feature_columns].values
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.pipeline is None:
            raise ValueError("Model must be trained first")
        
        # Get the trained model from pipeline
        model = self.pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            # Get selected features
            feature_selector = self.pipeline.named_steps['feature_selection']
            selected_features = feature_selector.get_support()
            selected_feature_names = [self.feature_columns[i] for i in range(len(selected_features)) if selected_features[i]]
            
            # Get importance scores
            importances = model.feature_importances_
            
            feature_importance = dict(zip(selected_feature_names, importances))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        else:
            logger.warning("Model does not support feature importance")
            return {}
    
    def evaluate_model(self, test_data_path: str = None, X_test: np.ndarray = None, y_test: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")
        
        if X_test is None or y_test is None:
            if test_data_path is None:
                raise ValueError("Either test_data_path or X_test/y_test must be provided")
            
            X_test, y_test = self.load_training_data(test_data_path)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,  # Mean Absolute Percentage Error
            'num_samples': len(y_test)
        }
        
        # Prediction accuracy within thresholds
        for threshold in [10, 20, 50]:
            within_threshold = np.abs(y_test - y_pred) <= threshold
            metrics[f'accuracy_within_{threshold}'] = np.mean(within_threshold)
        
        return metrics
    
    def save_model(self, model_path: str = 'models/improved_rul.pkl'):
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_engineer': self.feature_engineer,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'cv_scores': self.cv_scores
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'models/improved_rul.pkl'):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.cv_scores = model_data.get('cv_scores', None)
        
        logger.info(f"Model loaded from {model_path}")

class ModelComparison:
    """Compare different RUL prediction models"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = {}
    
    def compare_models(self, model_types: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Compare performance of different model types"""
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting']
        
        logger.info(f"Comparing models: {model_types}")
        
        for model_type in model_types:
            logger.info(f"Training {model_type}...")
            
            try:
                predictor = ImprovedRULPredictor(model_type=model_type)
                training_metrics = predictor.train_model(self.data_path, tune_hyperparameters=False)
                
                self.results[model_type] = {
                    'training_metrics': training_metrics,
                    'feature_importance': predictor.get_feature_importance()
                }
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                self.results[model_type] = {'error': str(e)}
        
        # Find best model
        best_model = min(self.results.keys(), 
                        key=lambda x: self.results[x].get('training_metrics', {}).get('rmse', float('inf')))
        
        logger.info(f"Best model: {best_model}")
        
        return self.results

if __name__ == "__main__":
    # Example usage
    try:
        # Single model training
        predictor = ImprovedRULPredictor(model_type='random_forest')
        training_results = predictor.train_model(tune_hyperparameters=True)
        
        print(f"Training results: {training_results}")
        
        # Feature importance
        importance = predictor.get_feature_importance()
        print(f"Top 10 features: {list(importance.items())[:10]}")
        
        # Save model
        predictor.save_model()
        
        # Model comparison
        comparison = ModelComparison('data/processed/train_processed.csv')
        results = comparison.compare_models()
        
        print("\nModel comparison results:")
        for model_type, result in results.items():
            if 'training_metrics' in result:
                metrics = result['training_metrics']
                print(f"{model_type}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise