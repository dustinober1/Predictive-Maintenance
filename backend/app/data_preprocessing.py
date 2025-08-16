import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreprocessor:
    """Data preprocessing pipeline for NASA Turbofan Engine dataset"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None
        self.sensor_columns = None
        self.setting_columns = None
        
    def load_data(self, data_path='data/raw'):
        """Load raw NASA dataset files"""
        
        # Define column names
        columns = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        
        # Load training data
        train_df = pd.read_csv(f'{data_path}/train_FD001.txt', sep=' ', names=columns, index_col=False)
        
        # Load test data
        test_df = pd.read_csv(f'{data_path}/test_FD001.txt', sep=' ', names=columns, index_col=False)
        
        # Load RUL truth
        rul_df = pd.read_csv(f'{data_path}/RUL_FD001.txt', names=['RUL'], index_col=False)
        
        self.columns = columns
        self.sensor_columns = [col for col in columns if col.startswith('sensor_')]
        self.setting_columns = [col for col in columns if col.startswith('setting_')]
        
        return train_df, test_df, rul_df
    
    def calculate_rul(self, df):
        """Calculate Remaining Useful Life for training data"""
        df_copy = df.copy()
        df_copy['RUL'] = df_copy.groupby('engine_id')['cycle'].transform('max') - df_copy['cycle']
        return df_copy
    
    def add_rolling_features(self, df, windows=[5, 10, 20]):
        """Add rolling statistical features"""
        df_features = df.copy()
        
        for window in windows:
            for sensor in self.sensor_columns:
                # Rolling mean
                df_features[f'{sensor}_rolling_mean_{window}'] = df_features.groupby('engine_id')[sensor].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
                
                # Rolling std
                df_features[f'{sensor}_rolling_std_{window}'] = df_features.groupby('engine_id')[sensor].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                
                # Rolling max
                df_features[f'{sensor}_rolling_max_{window}'] = df_features.groupby('engine_id')[sensor].rolling(window, min_periods=1).max().reset_index(0, drop=True)
                
                # Rolling min
                df_features[f'{sensor}_rolling_min_{window}'] = df_features.groupby('engine_id')[sensor].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        
        # Fill NaN values with 0
        df_features = df_features.fillna(0)
        
        return df_features
    
    def add_degradation_features(self, df):
        """Add degradation-related features"""
        df_deg = df.copy()
        
        # Cycle-based features
        df_deg['cycle_norm'] = df_deg.groupby('engine_id')['cycle'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        
        # Sensor trend features (difference from first cycle)
        for sensor in self.sensor_columns:
            df_deg[f'{sensor}_trend'] = df_deg.groupby('engine_id')[sensor].transform(lambda x: x - x.iloc[0])
        
        # Sensor rate of change
        for sensor in self.sensor_columns:
            df_deg[f'{sensor}_diff'] = df_deg.groupby('engine_id')[sensor].diff().fillna(0)
        
        return df_deg
    
    def preprocess_features(self, df, fit_scaler=True):
        """Preprocess features for ML models"""
        
        # Select feature columns (exclude engine_id, cycle, RUL)
        feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL']]
        
        X = df[feature_cols].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    def create_sequences(self, df, sequence_length=50, target_col='RUL'):
        """Create sequences for LSTM model"""
        sequences = []
        targets = []
        
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
            
            # Feature columns
            feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL']]
            engine_features = engine_data[feature_cols].values
            
            if target_col in engine_data.columns:
                engine_targets = engine_data[target_col].values
            else:
                engine_targets = None
            
            # Create sequences
            for i in range(len(engine_features) - sequence_length + 1):
                seq = engine_features[i:i + sequence_length]
                sequences.append(seq)
                
                if engine_targets is not None:
                    targets.append(engine_targets[i + sequence_length - 1])
        
        sequences = np.array(sequences)
        targets = np.array(targets) if targets else None
        
        return sequences, targets
    
    def prepare_anomaly_detection_data(self, df, healthy_threshold=0.8):
        """Prepare data for anomaly detection (healthy vs degraded)"""
        
        # Consider data as "healthy" if RUL > threshold * max_RUL for each engine
        df_anomaly = df.copy()
        
        # Calculate max RUL for each engine
        max_rul_per_engine = df_anomaly.groupby('engine_id')['RUL'].max()
        df_anomaly['max_rul'] = df_anomaly['engine_id'].map(max_rul_per_engine)
        
        # Label as healthy (0) or degraded (1)
        df_anomaly['is_anomaly'] = (df_anomaly['RUL'] / df_anomaly['max_rul'] < (1 - healthy_threshold)).astype(int)
        
        return df_anomaly
    
    def process_training_data(self, save_path='data/processed'):
        """Complete preprocessing pipeline for training data"""
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        # Load raw data
        train_df, test_df, rul_df = self.load_data()
        
        print(f"Raw training data shape: {train_df.shape}")
        print(f"Raw test data shape: {test_df.shape}")
        
        # Calculate RUL for training data
        train_df = self.calculate_rul(train_df)
        
        # Add features
        print("Adding rolling features...")
        train_df = self.add_rolling_features(train_df)
        
        print("Adding degradation features...")
        train_df = self.add_degradation_features(train_df)
        
        # Preprocess features
        print("Preprocessing features...")
        feature_cols = [col for col in train_df.columns if col not in ['engine_id', 'cycle', 'RUL']]
        train_features = self.preprocess_features(train_df[['engine_id', 'cycle'] + feature_cols + ['RUL']], fit_scaler=True)
        
        # Combine with metadata
        train_processed = pd.concat([
            train_df[['engine_id', 'cycle', 'RUL']].reset_index(drop=True),
            train_features.reset_index(drop=True)
        ], axis=1)
        
        # Prepare anomaly detection data
        print("Preparing anomaly detection data...")
        train_anomaly = self.prepare_anomaly_detection_data(train_processed)
        
        # Create sequences for LSTM
        print("Creating sequences for LSTM...")
        X_seq, y_seq = self.create_sequences(train_processed, sequence_length=30, target_col='RUL')
        
        # Save processed data
        train_processed.to_csv(f'{save_path}/train_processed.csv', index=False)
        train_anomaly.to_csv(f'{save_path}/train_anomaly.csv', index=False)
        
        # Save sequences
        np.save(f'{save_path}/X_sequences.npy', X_seq)
        np.save(f'{save_path}/y_sequences.npy', y_seq)
        
        # Save scaler
        with open(f'{save_path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open(f'{save_path}/feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print(f"Processed training data shape: {train_processed.shape}")
        print(f"Sequence data shape: {X_seq.shape}")
        print(f"Healthy samples: {(train_anomaly['is_anomaly'] == 0).sum()}")
        print(f"Degraded samples: {(train_anomaly['is_anomaly'] == 1).sum()}")
        
        return train_processed, train_anomaly, X_seq, y_seq
    
    def process_test_data(self, test_df, rul_df, save_path='data/processed'):
        """Process test data using fitted preprocessor"""
        
        # Load saved scaler and feature columns
        with open(f'{save_path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{save_path}/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Add features to test data
        test_df = self.add_rolling_features(test_df)
        test_df = self.add_degradation_features(test_df)
        
        # Preprocess features
        test_features = self.preprocess_features(test_df[['engine_id', 'cycle'] + feature_cols], fit_scaler=False)
        
        # Combine with metadata
        test_processed = pd.concat([
            test_df[['engine_id', 'cycle']].reset_index(drop=True),
            test_features.reset_index(drop=True)
        ], axis=1)
        
        # Create sequences for LSTM
        X_test_seq, _ = self.create_sequences(test_processed, sequence_length=30, target_col=None)
        
        return test_processed, X_test_seq

if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    preprocessor.process_training_data()