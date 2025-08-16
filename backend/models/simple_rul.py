import csv
import math
import random
import json
import os

class SimpleRULPredictor:
    """Simple RUL predictor using linear regression with feature engineering"""
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_indices = None
        self.learning_rate = 0.001
        
    def load_training_data(self, filepath='data/processed/train_processed.csv'):
        """Load training data"""
        X = []
        y = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Find feature column indices
            feature_start = 2  # Skip engine_id and cycle
            rul_idx = header.index('RUL') if 'RUL' in header else 26
            cycle_idx = header.index('cycle')
            
            # Use original sensor features plus cycle information
            sensor_indices = [i for i in range(feature_start, rul_idx) if 'sensor_' in header[i] and 'trend' not in header[i]]
            setting_indices = [i for i in range(feature_start, rul_idx) if 'setting_' in header[i]]
            
            self.feature_indices = [cycle_idx] + setting_indices + sensor_indices
            
            print(f"Using {len(self.feature_indices)} features for RUL prediction")
            
            for row in reader:
                if len(row) < len(header):
                    continue
                
                # Extract features
                features = [float(row[i]) for i in self.feature_indices]
                rul = float(row[rul_idx])
                
                # Add engineered features
                cycle = float(row[cycle_idx])
                features.extend([
                    cycle ** 0.5,  # Square root of cycle
                    math.log(cycle + 1),  # Log of cycle
                    1 / (cycle + 1),  # Inverse cycle
                ])
                
                X.append(features)
                y.append(rul)
        
        print(f"Loaded {len(X)} samples for training")
        return X, y
    
    def train_model(self, data_path='data/processed/train_processed.csv', epochs=100):
        """Train the RUL prediction model using gradient descent"""
        
        # Load data
        X, y = self.load_training_data(data_path)
        
        if not X:
            raise ValueError("No training data found")
        
        # Initialize weights
        num_features = len(X[0])
        self.weights = [random.gauss(0, 0.1) for _ in range(num_features)]
        self.bias = 0.0
        
        print(f"Training RUL predictor with {num_features} features...")
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            combined = list(zip(X, y))
            random.shuffle(combined)
            X_shuffled, y_shuffled = zip(*combined)
            
            for i in range(len(X_shuffled)):
                # Forward pass
                prediction = self.predict_single(X_shuffled[i])
                
                # Compute loss (MSE)
                error = prediction - y_shuffled[i]
                total_loss += error ** 2
                
                # Backward pass (gradient descent)
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * error * X_shuffled[i][j]
                
                # Update bias
                self.bias -= self.learning_rate * error
            
            avg_loss = total_loss / len(X)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("RUL training completed!")
        return self.weights, self.bias
    
    def predict_single(self, features):
        """Predict RUL for a single sample"""
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        prediction = sum(self.weights[i] * features[i] for i in range(len(features))) + self.bias
        return max(0, prediction)  # RUL should be non-negative
    
    def predict(self, X):
        """Predict RUL for multiple samples"""
        return [self.predict_single(x) for x in X]
    
    def evaluate_model(self, test_data_path=None, X_test=None, y_test=None):
        """Evaluate model performance"""
        if X_test is None or y_test is None:
            if test_data_path:
                X_test, y_test = self.load_training_data(test_data_path)
            else:
                raise ValueError("Need test data for evaluation")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = sum((predictions[i] - y_test[i]) ** 2 for i in range(len(predictions))) / len(predictions)
        rmse = math.sqrt(mse)
        mae = sum(abs(predictions[i] - y_test[i]) for i in range(len(predictions))) / len(predictions)
        
        # R-squared
        y_mean = sum(y_test) / len(y_test)
        ss_tot = sum((y_test[i] - y_mean) ** 2 for i in range(len(y_test)))
        ss_res = sum((y_test[i] - predictions[i]) ** 2 for i in range(len(y_test)))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'num_samples': len(predictions)
        }
    
    def save_model(self, model_path='models/simple_rul.json'):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.weights is None:
            raise ValueError("No model to save")
        
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_indices': self.feature_indices,
            'learning_rate': self.learning_rate
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f)
        
        print(f"RUL model saved to {model_path}")
    
    def load_model(self, model_path='models/simple_rul.json'):
        """Load trained model"""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.feature_indices = model_data['feature_indices']
        self.learning_rate = model_data['learning_rate']
        
        print(f"RUL model loaded from {model_path}")

if __name__ == "__main__":
    # Train and evaluate RUL prediction model
    predictor = SimpleRULPredictor()
    
    # Train model
    predictor.train_model(epochs=50)
    
    # Load data for evaluation (using same data for demo)
    X_test, y_test = predictor.load_training_data()
    
    # Use a subset for evaluation to avoid overfitting metrics
    test_size = min(1000, len(X_test))
    X_eval = X_test[:test_size]
    y_eval = y_test[:test_size]
    
    # Evaluate
    metrics = predictor.evaluate_model(X_test=X_eval, y_test=y_eval)
    print(f"\nRUL Model Performance:")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"R-squared: {metrics['r_squared']:.3f}")
    
    # Save model
    predictor.save_model()