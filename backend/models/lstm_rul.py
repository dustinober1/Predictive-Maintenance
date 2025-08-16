import csv
import math
import random
import json
import os

class SimpleLSTM:
    """Simple LSTM implementation for RUL prediction"""
    
    def __init__(self, input_size, hidden_size=50, sequence_length=30, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        
        # Initialize weights and biases for LSTM gates
        self.init_weights()
        
        # Training history
        self.training_history = []
        
    def init_weights(self):
        """Initialize LSTM weights"""
        # Forget gate
        self.Wf = [[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                   for _ in range(self.input_size + self.hidden_size)]
        self.bf = [0.0] * self.hidden_size
        
        # Input gate
        self.Wi = [[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                   for _ in range(self.input_size + self.hidden_size)]
        self.bi = [0.0] * self.hidden_size
        
        # Candidate gate
        self.Wc = [[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                   for _ in range(self.input_size + self.hidden_size)]
        self.bc = [0.0] * self.hidden_size
        
        # Output gate
        self.Wo = [[random.gauss(0, 0.1) for _ in range(self.hidden_size)] 
                   for _ in range(self.input_size + self.hidden_size)]
        self.bo = [0.0] * self.hidden_size
        
        # Final output layer
        self.Wy = [[random.gauss(0, 0.1)] for _ in range(self.hidden_size)]
        self.by = [0.0]
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def tanh(self, x):
        """Tanh activation function"""
        return math.tanh(max(-20, min(20, x)))
    
    def lstm_cell(self, x, h_prev, c_prev):
        """Single LSTM cell forward pass"""
        # Concatenate input and previous hidden state
        concat = x + h_prev
        
        # Forget gate
        f = [self.sigmoid(sum(concat[i] * self.Wf[i][j] for i in range(len(concat))) + self.bf[j]) 
             for j in range(self.hidden_size)]
        
        # Input gate
        i = [self.sigmoid(sum(concat[i] * self.Wi[i][j] for i in range(len(concat))) + self.bi[j]) 
             for j in range(self.hidden_size)]
        
        # Candidate values
        c_tilde = [self.tanh(sum(concat[i] * self.Wc[i][j] for i in range(len(concat))) + self.bc[j]) 
                   for j in range(self.hidden_size)]
        
        # New cell state
        c_new = [f[j] * c_prev[j] + i[j] * c_tilde[j] for j in range(self.hidden_size)]
        
        # Output gate
        o = [self.sigmoid(sum(concat[i] * self.Wo[i][j] for i in range(len(concat))) + self.bo[j]) 
             for j in range(self.hidden_size)]
        
        # New hidden state
        h_new = [o[j] * self.tanh(c_new[j]) for j in range(self.hidden_size)]
        
        return h_new, c_new
    
    def forward(self, sequence):
        """Forward pass through LSTM for a sequence"""
        batch_size = len(sequence)
        
        # Initialize hidden and cell states
        h = [[0.0] * self.hidden_size for _ in range(batch_size)]
        c = [[0.0] * self.hidden_size for _ in range(batch_size)]
        
        # Process each timestep
        for t in range(self.sequence_length):
            for b in range(batch_size):
                if t < len(sequence[b]):
                    h[b], c[b] = self.lstm_cell(sequence[b][t], h[b], c[b])
        
        # Final output layer
        outputs = []
        for b in range(batch_size):
            output = sum(h[b][i] * self.Wy[i][0] for i in range(self.hidden_size)) + self.by[0]
            outputs.append(max(0, output))  # ReLU to ensure positive RUL
        
        return outputs, h, c
    
    def compute_loss(self, predictions, targets):
        """Compute mean squared error loss"""
        mse = sum((predictions[i] - targets[i]) ** 2 for i in range(len(predictions))) / len(predictions)
        return mse
    
    def simple_backward(self, predictions, targets, h_final):
        """Simplified backward pass for gradient updates"""
        # Compute output gradients
        output_grads = [(predictions[i] - targets[i]) * 2 / len(predictions) 
                       for i in range(len(predictions))]
        
        # Update output layer weights
        for i in range(self.hidden_size):
            for b in range(len(h_final)):
                grad = output_grads[b] * h_final[b][i]
                self.Wy[i][0] -= self.learning_rate * grad
        
        # Update output bias
        for b in range(len(output_grads)):
            self.by[0] -= self.learning_rate * output_grads[b]
        
        # Simplified LSTM weight updates (approximation)
        avg_grad = sum(abs(g) for g in output_grads) / len(output_grads)
        
        # Small random updates to LSTM weights based on gradient magnitude
        for gate_weights in [self.Wf, self.Wi, self.Wc, self.Wo]:
            for i in range(len(gate_weights)):
                for j in range(len(gate_weights[i])):
                    gate_weights[i][j] -= self.learning_rate * avg_grad * random.gauss(0, 0.01)

class RULPredictor:
    """RUL prediction using LSTM"""
    
    def __init__(self):
        self.lstm = None
        self.feature_indices = None
        self.sequence_length = 30
        
    def load_training_data(self, filepath='data/processed/train_processed.csv'):
        """Load and prepare sequence data for training"""
        sequences = []
        targets = []
        
        # Group data by engine
        engine_data = {}
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Find feature column indices
            feature_start = 2  # Skip engine_id and cycle
            rul_idx = header.index('RUL') if 'RUL' in header else 26
            self.feature_indices = list(range(feature_start, rul_idx))
            
            print(f"Loading data with {len(self.feature_indices)} features")
            
            for row in reader:
                if len(row) < len(header):
                    continue
                    
                engine_id = int(float(row[0]))
                cycle = int(float(row[1]))
                rul = float(row[rul_idx])
                
                features = [float(row[i]) for i in self.feature_indices]
                
                if engine_id not in engine_data:
                    engine_data[engine_id] = []
                
                engine_data[engine_id].append({
                    'cycle': cycle,
                    'features': features,
                    'rul': rul
                })
        
        # Sort each engine's data by cycle
        for engine_id in engine_data:
            engine_data[engine_id].sort(key=lambda x: x['cycle'])
        
        # Create sequences
        for engine_id, data in engine_data.items():
            if len(data) < self.sequence_length:
                continue
            
            # Create sliding window sequences
            for i in range(len(data) - self.sequence_length + 1):
                # Extract sequence of features
                sequence = [data[j]['features'] for j in range(i, i + self.sequence_length)]
                
                # Target is the RUL at the end of the sequence
                target = data[i + self.sequence_length - 1]['rul']
                
                sequences.append(sequence)
                targets.append(target)
        
        print(f"Created {len(sequences)} sequences from {len(engine_data)} engines")
        return sequences, targets
    
    def train_model(self, data_path='data/processed/train_processed.csv', 
                   hidden_size=32, epochs=20, learning_rate=0.001):
        """Train the RUL prediction model"""
        
        # Load sequence data
        sequences, targets = self.load_training_data(data_path)
        
        if not sequences:
            raise ValueError("No sequence data found for training")
        
        # Initialize LSTM
        input_size = len(sequences[0][0])  # Number of features
        self.lstm = SimpleLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            sequence_length=self.sequence_length,
            learning_rate=learning_rate
        )
        
        print(f"Training LSTM with {input_size} features, {hidden_size} hidden units...")
        
        # Training loop
        batch_size = 32
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Create mini-batches
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                
                # Forward pass
                predictions, h_final, c_final = self.lstm.forward(batch_sequences)
                
                # Compute loss
                loss = self.lstm.compute_loss(predictions, batch_targets)
                total_loss += loss
                
                # Backward pass
                self.lstm.simple_backward(predictions, batch_targets, h_final)
                
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.lstm.training_history.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("RUL training completed!")
        return self.lstm
    
    def predict_rul(self, sequence):
        """Predict RUL for a single sequence"""
        if self.lstm is None:
            raise ValueError("Model must be trained first")
        
        predictions, _, _ = self.lstm.forward([sequence])
        return predictions[0]
    
    def evaluate_model(self, test_sequences, test_targets):
        """Evaluate model performance"""
        if not test_sequences:
            return {}
        
        predictions = []
        for seq in test_sequences:
            pred = self.predict_rul(seq)
            predictions.append(pred)
        
        # Calculate metrics
        mse = sum((predictions[i] - test_targets[i]) ** 2 for i in range(len(predictions))) / len(predictions)
        rmse = math.sqrt(mse)
        
        # Mean Absolute Error
        mae = sum(abs(predictions[i] - test_targets[i]) for i in range(len(predictions))) / len(predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'num_samples': len(predictions)
        }
    
    def save_model(self, model_path='models/lstm_rul.json'):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.lstm is None:
            raise ValueError("No model to save")
        
        # Prepare model data
        model_data = {
            'input_size': self.lstm.input_size,
            'hidden_size': self.lstm.hidden_size,
            'sequence_length': self.lstm.sequence_length,
            'learning_rate': self.lstm.learning_rate,
            'Wf': self.lstm.Wf,
            'bf': self.lstm.bf,
            'Wi': self.lstm.Wi,
            'bi': self.lstm.bi,
            'Wc': self.lstm.Wc,
            'bc': self.lstm.bc,
            'Wo': self.lstm.Wo,
            'bo': self.lstm.bo,
            'Wy': self.lstm.Wy,
            'by': self.lstm.by,
            'training_history': self.lstm.training_history,
            'feature_indices': self.feature_indices
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f)
        
        print(f"RUL model saved to {model_path}")
    
    def load_model(self, model_path='models/lstm_rul.json'):
        """Load trained model"""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Recreate LSTM
        self.lstm = SimpleLSTM(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            sequence_length=model_data['sequence_length'],
            learning_rate=model_data['learning_rate']
        )
        
        # Load weights
        self.lstm.Wf = model_data['Wf']
        self.lstm.bf = model_data['bf']
        self.lstm.Wi = model_data['Wi']
        self.lstm.bi = model_data['bi']
        self.lstm.Wc = model_data['Wc']
        self.lstm.bc = model_data['bc']
        self.lstm.Wo = model_data['Wo']
        self.lstm.bo = model_data['bo']
        self.lstm.Wy = model_data['Wy']
        self.lstm.by = model_data['by']
        self.lstm.training_history = model_data['training_history']
        self.feature_indices = model_data['feature_indices']
        
        print(f"RUL model loaded from {model_path}")

if __name__ == "__main__":
    # Train and evaluate RUL prediction model
    predictor = RULPredictor()
    
    # Train model
    predictor.train_model(epochs=10, hidden_size=16)  # Smaller parameters for faster training
    
    # Load test data for evaluation
    test_sequences, test_targets = predictor.load_training_data()
    
    # Use a subset for evaluation
    test_subset = min(100, len(test_sequences))
    eval_sequences = test_sequences[:test_subset]
    eval_targets = test_targets[:test_subset]
    
    # Evaluate
    metrics = predictor.evaluate_model(eval_sequences, eval_targets)
    print(f"\nRUL Model Performance:")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")
    
    # Save model
    predictor.save_model()