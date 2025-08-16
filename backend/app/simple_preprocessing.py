import csv
import math
import os

class SimpleDataPreprocessor:
    """Simplified data preprocessing without external dependencies"""
    
    def __init__(self):
        self.feature_means = {}
        self.feature_stds = {}
        
    def load_data(self, data_path='data/raw'):
        """Load raw data from CSV files"""
        
        # Load training data
        train_data = []
        with open(f'{data_path}/train_FD001.txt', 'r') as f:
            for line in f:
                values = line.strip().split()
                train_data.append([float(v) for v in values])
        
        # Load test data
        test_data = []
        with open(f'{data_path}/test_FD001.txt', 'r') as f:
            for line in f:
                values = line.strip().split()
                test_data.append([float(v) for v in values])
        
        # Load RUL truth
        rul_data = []
        with open(f'{data_path}/RUL_FD001.txt', 'r') as f:
            for line in f:
                rul_data.append(float(line.strip()))
        
        return train_data, test_data, rul_data
    
    def calculate_rul(self, data):
        """Calculate Remaining Useful Life for training data"""
        # Group by engine_id and calculate max cycle for each engine
        engine_max_cycles = {}
        
        for row in data:
            engine_id = int(row[0])
            cycle = int(row[1])
            
            if engine_id not in engine_max_cycles:
                engine_max_cycles[engine_id] = cycle
            else:
                engine_max_cycles[engine_id] = max(engine_max_cycles[engine_id], cycle)
        
        # Add RUL to each row
        data_with_rul = []
        for row in data:
            engine_id = int(row[0])
            cycle = int(row[1])
            rul = engine_max_cycles[engine_id] - cycle
            data_with_rul.append(row + [rul])
        
        return data_with_rul
    
    def calculate_statistics(self, data, start_col=2, end_col=25):
        """Calculate mean and std for features"""
        feature_values = {}
        
        # Collect all values for each feature
        for row in data:
            for i in range(start_col, min(end_col, len(row))):
                if i not in feature_values:
                    feature_values[i] = []
                feature_values[i].append(row[i])
        
        # Calculate mean and std
        for col, values in feature_values.items():
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1.0
            
            self.feature_means[col] = mean
            self.feature_stds[col] = std
    
    def normalize_features(self, data, start_col=2, end_col=25):
        """Normalize features using calculated statistics"""
        normalized_data = []
        
        for row in data:
            new_row = row[:start_col]  # Keep engine_id and cycle
            
            for i in range(start_col, min(end_col, len(row))):
                if i in self.feature_means and self.feature_stds[i] > 0:
                    normalized_value = (row[i] - self.feature_means[i]) / self.feature_stds[i]
                else:
                    normalized_value = row[i]
                new_row.append(normalized_value)
            
            # Keep remaining columns (like RUL)
            if len(row) > end_col:
                new_row.extend(row[end_col:])
            
            normalized_data.append(new_row)
        
        return normalized_data
    
    def add_simple_features(self, data):
        """Add simple engineered features"""
        enhanced_data = []
        
        # Group data by engine for rolling calculations
        engine_data = {}
        for row in data:
            engine_id = int(row[0])
            if engine_id not in engine_data:
                engine_data[engine_id] = []
            engine_data[engine_id].append(row)
        
        # Process each engine
        for engine_id, engine_rows in engine_data.items():
            # Sort by cycle
            engine_rows.sort(key=lambda x: x[1])
            
            for i, row in enumerate(engine_rows):
                new_row = row.copy()
                
                # Add cycle normalization
                max_cycle = max(r[1] for r in engine_rows)
                cycle_norm = row[1] / max_cycle if max_cycle > 0 else 0
                new_row.append(cycle_norm)
                
                # Add simple trend features (difference from first reading)
                if i == 0:
                    for j in range(2, min(25, len(row))):
                        new_row.append(0)  # No trend for first cycle
                else:
                    first_row = engine_rows[0]
                    for j in range(2, min(25, len(row))):
                        trend = row[j] - first_row[j]
                        new_row.append(trend)
                
                enhanced_data.append(new_row)
        
        return enhanced_data
    
    def create_anomaly_labels(self, data, threshold=0.3):
        """Create anomaly labels based on RUL"""
        labeled_data = []
        
        for row in data:
            if len(row) < 3:  # Need at least engine_id, cycle, and some features
                continue
                
            # Assume RUL is the last column
            rul = row[-1] if len(row) > 2 else 0
            
            # Calculate max RUL for this engine
            engine_id = int(row[0])
            
            # Simple heuristic: anomaly if RUL is in bottom threshold of lifecycle
            is_anomaly = 1 if rul < 50 else 0  # Simple threshold
            
            new_row = row + [is_anomaly]
            labeled_data.append(new_row)
        
        return labeled_data
    
    def process_and_save(self, save_path='data/processed'):
        """Complete preprocessing pipeline"""
        
        os.makedirs(save_path, exist_ok=True)
        
        print("Loading raw data...")
        train_data, test_data, rul_data = self.load_data()
        
        print(f"Raw training data: {len(train_data)} rows")
        print(f"Raw test data: {len(test_data)} rows")
        
        # Calculate RUL for training data
        print("Calculating RUL...")
        train_data = self.calculate_rul(train_data)
        
        # Add features
        print("Adding features...")
        train_data = self.add_simple_features(train_data)
        
        # Calculate normalization statistics
        print("Calculating normalization statistics...")
        self.calculate_statistics(train_data)
        
        # Normalize features
        print("Normalizing features...")
        train_data = self.normalize_features(train_data)
        
        # Create anomaly labels
        print("Creating anomaly labels...")
        train_anomaly = self.create_anomaly_labels(train_data)
        
        # Save processed training data
        with open(f'{save_path}/train_processed.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
            header += ['RUL', 'cycle_norm'] + [f'sensor_{i}_trend' for i in range(1, 22)]
            writer.writerow(header)
            
            # Write data
            for row in train_data:
                writer.writerow(row)
        
        # Save anomaly data
        with open(f'{save_path}/train_anomaly.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
            header += ['RUL', 'cycle_norm'] + [f'sensor_{i}_trend' for i in range(1, 22)] + ['is_anomaly']
            writer.writerow(header)
            
            # Write data
            for row in train_anomaly:
                writer.writerow(row)
        
        # Save statistics for later use
        with open(f'{save_path}/normalization_stats.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature_index', 'mean', 'std'])
            for col in self.feature_means:
                writer.writerow([col, self.feature_means[col], self.feature_stds[col]])
        
        print(f"Processed training data: {len(train_data)} rows")
        print(f"Anomaly training data: {len(train_anomaly)} rows")
        print("Data preprocessing completed!")
        
        return train_data, train_anomaly

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = SimpleDataPreprocessor()
    preprocessor.process_and_save()