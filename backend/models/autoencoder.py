import csv
import math
import random
import json
import os

class SimpleAutoencoder:
    """Simple autoencoder implementation for anomaly detection"""
    
    def __init__(self, input_size, hidden_size=32, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.init_weights()
        
        # Training history
        self.training_history = []
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        # Encoder weights
        self.W1 = [[random.gauss(0, math.sqrt(2.0 / (self.input_size + self.hidden_size))) 
                   for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.b1 = [0.0] * self.hidden_size
        
        # Decoder weights
        self.W2 = [[random.gauss(0, math.sqrt(2.0 / (self.hidden_size + self.input_size))) 
                   for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.b2 = [0.0] * self.input_size
    
    def relu(self, x):
        """ReLU activation function"""
        return max(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return 1 if x > 0 else 0
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        # Encoder
        z1 = [sum(x[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j] 
              for j in range(self.hidden_size)]
        h = [self.relu(z) for z in z1]
        
        # Decoder
        z2 = [sum(h[i] * self.W2[i][j] for i in range(self.hidden_size)) + self.b2[j] 
              for j in range(self.input_size)]
        output = z2  # Linear output for reconstruction
        
        return output, h, z1
    
    def compute_loss(self, x, x_reconstructed):
        """Compute mean squared error loss"""
        mse = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x))) / len(x)
        return mse
    
    def backward(self, x, x_reconstructed, h, z1):
        """Backward pass and weight updates"""
        # Output layer gradients
        output_grad = [(x_reconstructed[i] - x[i]) * 2 / self.input_size 
                      for i in range(self.input_size)]
        
        # Hidden layer gradients
        hidden_grad = [sum(output_grad[j] * self.W2[i][j] for j in range(self.input_size)) 
                      for i in range(self.hidden_size)]
        
        # Apply ReLU derivative to hidden gradients
        hidden_grad = [hidden_grad[i] * self.relu_derivative(z1[i]) 
                      for i in range(self.hidden_size)]
        
        # Update decoder weights and biases
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.W2[i][j] -= self.learning_rate * output_grad[j] * h[i]
        
        for j in range(self.input_size):
            self.b2[j] -= self.learning_rate * output_grad[j]
        
        # Update encoder weights and biases
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W1[i][j] -= self.learning_rate * hidden_grad[j] * x[i]
        
        for j in range(self.hidden_size):
            self.b1[j] -= self.learning_rate * hidden_grad[j]
    
    def train_epoch(self, X_train):
        """Train for one epoch"""
        total_loss = 0
        
        for x in X_train:
            # Forward pass
            x_reconstructed, h, z1 = self.forward(x)
            
            # Compute loss
            loss = self.compute_loss(x, x_reconstructed)
            total_loss += loss
            
            # Backward pass
            self.backward(x, x_reconstructed, h, z1)
        
        avg_loss = total_loss / len(X_train)
        return avg_loss
    
    def train(self, X_train, epochs=100, verbose=True):
        """Train the autoencoder"""
        if verbose:
            print(f"Training autoencoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(X_train)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print("Training completed!")
    
    def predict(self, x):
        """Predict reconstruction and compute reconstruction error"""
        x_reconstructed, _, _ = self.forward(x)
        reconstruction_error = self.compute_loss(x, x_reconstructed)
        return x_reconstructed, reconstruction_error
    
    def detect_anomaly(self, x, threshold):
        """Detect anomaly based on reconstruction error threshold"""
        _, reconstruction_error = self.predict(x)
        return reconstruction_error > threshold, reconstruction_error
    
    def save_model(self, filepath):
        """Save model weights to file"""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load model weights from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.learning_rate = model_data['learning_rate']
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.training_history = model_data.get('training_history', [])

class AnomalyDetector:
    """Main class for anomaly detection using autoencoder"""
    
    def __init__(self):
        self.autoencoder = None
        self.threshold = None
        self.feature_indices = None
        
    def load_training_data(self, filepath='data/processed/train_anomaly.csv'):
        """Load preprocessed training data"""
        healthy_data = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Find feature column indices (excluding metadata)
            feature_start = 2  # Skip engine_id and cycle
            # Find RUL column (should be "RUL")
            rul_idx = header.index('RUL') if 'RUL' in header else 26
            # Features are from after engine_id/cycle to before RUL
            self.feature_indices = list(range(feature_start, rul_idx))
            
            print(f"Header length: {len(header)}, Feature indices: {feature_start} to {rul_idx}, RUL at: {rul_idx}")
            
            # Find anomaly column index
            anomaly_idx = header.index('is_anomaly') if 'is_anomaly' in header else -1
            
            for row in reader:
                if len(row) < len(header):
                    continue
                is_anomaly = float(row[anomaly_idx])  # Anomaly column
                
                # Only use healthy samples for training (threshold 0.5)
                if is_anomaly < 0.5:
                    features = [float(row[i]) for i in self.feature_indices]
                    healthy_data.append(features)
        
        print(f"Loaded {len(healthy_data)} healthy samples for training")
        return healthy_data
    
    def train_model(self, data_path='data/processed/train_anomaly.csv', 
                   hidden_size=32, epochs=50, learning_rate=0.01):
        """Train the anomaly detection model"""
        
        # Load healthy data
        healthy_data = self.load_training_data(data_path)
        
        if not healthy_data:
            raise ValueError("No healthy data found for training")
        
        # Initialize autoencoder
        input_size = len(healthy_data[0])
        self.autoencoder = SimpleAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate
        )
        
        print(f"Training autoencoder with {input_size} features...")
        
        # Train the model
        self.autoencoder.train(healthy_data, epochs=epochs, verbose=True)
        
        # Calculate threshold based on training data reconstruction errors
        reconstruction_errors = []
        for sample in healthy_data:
            _, error = self.autoencoder.predict(sample)
            reconstruction_errors.append(error)
        
        # Set threshold as 95th percentile of training errors
        reconstruction_errors.sort()
        threshold_index = int(0.95 * len(reconstruction_errors))
        self.threshold = reconstruction_errors[threshold_index]
        
        print(f"Anomaly detection threshold set to: {self.threshold:.6f}")
        
        return self.autoencoder, self.threshold
    
    def detect_anomalies(self, test_data_path='data/processed/train_anomaly.csv'):
        """Detect anomalies in test data"""
        if self.autoencoder is None or self.threshold is None:
            raise ValueError("Model must be trained first")
        
        predictions = []
        
        with open(test_data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Find anomaly column index
            anomaly_idx = header.index('is_anomaly') if 'is_anomaly' in header else -1
            
            for row in reader:
                engine_id = int(float(row[0]))
                cycle = int(float(row[1]))
                actual_anomaly = 1 if float(row[anomaly_idx]) >= 0.5 else 0
                
                # Extract features
                features = [float(row[i]) for i in self.feature_indices]
                
                # Predict anomaly
                is_anomaly, error = self.autoencoder.detect_anomaly(features, self.threshold)
                
                predictions.append({
                    'engine_id': engine_id,
                    'cycle': cycle,
                    'reconstruction_error': error,
                    'predicted_anomaly': int(is_anomaly),
                    'actual_anomaly': actual_anomaly
                })
        
        return predictions
    
    def evaluate_model(self, predictions):
        """Evaluate model performance"""
        tp = sum(1 for p in predictions if p['predicted_anomaly'] == 1 and p['actual_anomaly'] == 1)
        tn = sum(1 for p in predictions if p['predicted_anomaly'] == 0 and p['actual_anomaly'] == 0)
        fp = sum(1 for p in predictions if p['predicted_anomaly'] == 1 and p['actual_anomaly'] == 0)
        fn = sum(1 for p in predictions if p['predicted_anomaly'] == 0 and p['actual_anomaly'] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def save_model(self, model_path='models/autoencoder_anomaly.json'):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.autoencoder is None:
            raise ValueError("No model to save")
        
        # Save autoencoder
        self.autoencoder.save_model(model_path)
        
        # Save additional metadata
        metadata_path = model_path.replace('.json', '_metadata.json')
        metadata = {
            'threshold': self.threshold,
            'feature_indices': self.feature_indices
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/autoencoder_anomaly.json'):
        """Load trained model"""
        # Load autoencoder
        self.autoencoder = SimpleAutoencoder(input_size=1, hidden_size=1)  # Temporary
        self.autoencoder.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.threshold = metadata['threshold']
        self.feature_indices = metadata['feature_indices']
        
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Train and evaluate anomaly detection model
    detector = AnomalyDetector()
    
    # Train model with smaller parameters for faster testing
    detector.train_model(epochs=10, hidden_size=16)
    
    # Detect anomalies
    predictions = detector.detect_anomalies()
    
    # Evaluate
    metrics = detector.evaluate_model(predictions)
    print(f"\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    
    # Save model
    detector.save_model()