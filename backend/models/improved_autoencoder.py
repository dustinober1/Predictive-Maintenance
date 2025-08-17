import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedAutoencoder:
    """Improved autoencoder implementation using TensorFlow/Keras"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.encoder = None
        self.scaler = RobustScaler()
        self.threshold = None
        self.history = None
        
    def _build_model(self):
        """Build the autoencoder architecture"""
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(64, activation='relu', name='encoder_1')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoder_2')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu', name='decoder_1')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear', name='decoder_2')(decoded)
        
        # Full autoencoder
        self.model = keras.Model(input_layer, decoded, name='autoencoder')
        
        # Encoder model (for dimensionality reduction)
        self.encoder = keras.Model(input_layer, encoded, name='encoder')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built autoencoder: {self.input_dim} -> {self.encoding_dim} -> {self.input_dim}")
        
    def fit(self, X: np.ndarray, validation_split: float = 0.2, epochs: int = 100, 
            batch_size: int = 32, early_stopping: bool = True, verbose: int = 1) -> Dict[str, Any]:
        """Train the autoencoder on healthy data"""
        
        logger.info(f"Training autoencoder on {X.shape[0]} samples...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model if not exists
        if self.model is None:
            self._build_model()
        
        # Callbacks
        callbacks = []
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            )
            callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        self.history = self.model.fit(
            X_scaled, X_scaled,  # Autoencoder learns to reconstruct input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        # Calculate reconstruction errors for threshold setting
        reconstruction_errors = self._calculate_reconstruction_errors(X_scaled)
        
        # Set threshold as 95th percentile of training reconstruction errors
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        logger.info(f"Training completed. Threshold set to: {self.threshold:.6f}")
        
        return {
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'threshold': self.threshold,
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def _calculate_reconstruction_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for input data"""
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies for input data"""
        if self.model is None or self.threshold is None:
            raise ValueError("Model must be trained before prediction")
        
        # Scale input data
        X_scaled = self.scaler.transform(X)
        
        # Calculate reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_errors(X_scaled)
        
        # Determine anomalies
        anomalies = reconstruction_errors > self.threshold
        
        return anomalies, reconstruction_errors
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get encoded representations of input data"""
        if self.encoder is None:
            raise ValueError("Model must be trained before getting embeddings")
        
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model architecture and weights
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'learning_rate': self.learning_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler"""
        # Load model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.threshold = metadata['threshold']
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.learning_rate = metadata['learning_rate']
        
        # Rebuild encoder
        input_layer = self.model.input
        encoded_layer = self.model.get_layer('encoder_2').output
        self.encoder = keras.Model(input_layer, encoded_layer, name='encoder')
        
        logger.info(f"Model loaded from {filepath}")

class ImprovedAnomalyDetector:
    """Improved anomaly detection system with robust data handling"""
    
    def __init__(self):
        self.autoencoder = None
        self.feature_columns = None
        self.data_validator = DataValidator()
        
    def load_training_data(self, filepath: str = 'data/processed/train_anomaly.csv') -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate training data"""
        try:
            logger.info(f"Loading training data from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Training data file not found: {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Validate data structure
            self.data_validator.validate_dataframe(df)
            
            # Extract features and labels
            feature_cols = [col for col in df.columns if col.startswith(('sensor_', 'setting_')) and 'trend' not in col]
            
            if not feature_cols:
                raise ValueError("No valid feature columns found in data")
            
            self.feature_columns = feature_cols
            
            # Get healthy samples only (is_anomaly < 0.5)
            if 'is_anomaly' in df.columns:
                healthy_mask = df['is_anomaly'] < 0.5
                X = df.loc[healthy_mask, feature_cols].values
                y = df.loc[healthy_mask, 'is_anomaly'].values
            else:
                logger.warning("No 'is_anomaly' column found, using all data as healthy")
                X = df[feature_cols].values
                y = np.zeros(len(X))
            
            # Validate features
            X_clean = self.data_validator.clean_features(X)
            
            logger.info(f"Loaded {X_clean.shape[0]} healthy samples with {X_clean.shape[1]} features")
            
            return X_clean, y
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def train_model(self, data_path: str = 'data/processed/train_anomaly.csv', 
                   encoding_dim: int = 32, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Train the improved anomaly detection model"""
        
        # Load training data
        X_train, _ = self.load_training_data(data_path)
        
        # Initialize autoencoder
        self.autoencoder = ImprovedAutoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=encoding_dim
        )
        
        # Train model
        training_results = self.autoencoder.fit(X_train, epochs=epochs, **kwargs)
        
        return training_results
    
    def detect_anomalies(self, test_data_path: str = None, X_test: np.ndarray = None) -> pd.DataFrame:
        """Detect anomalies in test data"""
        if self.autoencoder is None:
            raise ValueError("Model must be trained before anomaly detection")
        
        if X_test is None:
            if test_data_path is None:
                raise ValueError("Either test_data_path or X_test must be provided")
            
            # Load test data
            df = pd.read_csv(test_data_path)
            X_test = df[self.feature_columns].values
            
            # Clean test data
            X_test = self.data_validator.clean_features(X_test)
            
            # Get metadata if available
            metadata_cols = ['engine_id', 'cycle']
            metadata = df[metadata_cols] if all(col in df.columns for col in metadata_cols) else None
            
            # Get true labels if available
            y_true = df['is_anomaly'].values if 'is_anomaly' in df.columns else None
        else:
            X_test = self.data_validator.clean_features(X_test)
            metadata = None
            y_true = None
        
        # Predict anomalies
        anomalies, reconstruction_errors = self.autoencoder.predict_anomaly(X_test)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_anomaly': anomalies.astype(int),
            'reconstruction_error': reconstruction_errors,
            'anomaly_score': (reconstruction_errors / self.autoencoder.threshold).clip(0, 5)  # Normalized score
        })
        
        # Add metadata if available
        if metadata is not None:
            results = pd.concat([metadata.reset_index(drop=True), results], axis=1)
        
        # Add true labels if available
        if y_true is not None:
            results['actual_anomaly'] = (y_true >= 0.5).astype(int)
        
        return results
    
    def evaluate_model(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance"""
        if 'actual_anomaly' not in predictions_df.columns:
            logger.warning("No actual labels available for evaluation")
            return {}
        
        y_true = predictions_df['actual_anomaly'].values
        y_pred = predictions_df['predicted_anomaly'].values
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC if reconstruction errors are available
        if 'reconstruction_error' in predictions_df.columns:
            metrics['auc_roc'] = roc_auc_score(y_true, predictions_df['reconstruction_error'])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Model evaluation - Accuracy: {metrics['accuracy']:.3f}, "
                   f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        
        return metrics
    
    def save_model(self, model_path: str = 'models/improved_autoencoder.pkl'):
        """Save the trained model"""
        if self.autoencoder is None:
            raise ValueError("No model to save")
        
        self.autoencoder.save_model(model_path)
        
        # Save feature column info
        metadata_path = model_path.replace('.pkl', '_features.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({'feature_columns': self.feature_columns}, f)
        
        logger.info(f"Anomaly detector saved to {model_path}")
    
    def load_model(self, model_path: str = 'models/improved_autoencoder.pkl'):
        """Load a trained model"""
        # Load autoencoder
        self.autoencoder = ImprovedAutoencoder(input_dim=1, encoding_dim=1)  # Temporary
        self.autoencoder.load_model(model_path)
        
        # Load feature columns
        metadata_path = model_path.replace('.pkl', '_features.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_columns = metadata['feature_columns']
        
        logger.info(f"Anomaly detector loaded from {model_path}")

class DataValidator:
    """Data validation and cleaning utilities"""
    
    def __init__(self):
        self.expected_columns = ['engine_id', 'cycle', 'is_anomaly']
        
    def validate_dataframe(self, df: pd.DataFrame):
        """Validate DataFrame structure and content"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for basic required columns
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        # Check for feature columns
        feature_cols = [col for col in df.columns if col.startswith(('sensor_', 'setting_'))]
        if len(feature_cols) < 5:
            raise ValueError(f"Insufficient feature columns found: {len(feature_cols)}")
        
        logger.info(f"Data validation passed: {len(df)} rows, {len(feature_cols)} features")
    
    def clean_features(self, X: np.ndarray) -> np.ndarray:
        """Clean and validate feature array"""
        if X.size == 0:
            raise ValueError("Empty feature array")
        
        # Replace infinite values
        X = np.where(np.isinf(X), np.nan, X)
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found NaN values, replacing with column medians")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Check for constant columns
        constant_cols = np.where(np.std(X, axis=0) == 0)[0]
        if len(constant_cols) > 0:
            logger.warning(f"Found {len(constant_cols)} constant columns")
        
        return X

if __name__ == "__main__":
    # Example usage
    detector = ImprovedAnomalyDetector()
    
    try:
        # Train model
        training_results = detector.train_model(epochs=50, batch_size=64)
        print(f"Training results: {training_results}")
        
        # Test detection
        results = detector.detect_anomalies()
        print(f"Detection results shape: {results.shape}")
        
        # Evaluate if labels available
        if 'actual_anomaly' in results.columns:
            metrics = detector.evaluate_model(results)
            print(f"Evaluation metrics: {metrics}")
        
        # Save model
        detector.save_model()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise