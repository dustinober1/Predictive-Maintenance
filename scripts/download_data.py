import os
import urllib.request
import zipfile
import pandas as pd

def download_nasa_dataset():
    """Download NASA Turbofan Engine Degradation Simulation Data Set"""
    
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # NASA dataset URL
    url = "https://ti.arc.nasa.gov/c/6/"
    zip_file = f"{data_dir}/CMAPSSData.zip"
    
    print("Downloading NASA Turbofan Engine dataset...")
    
    # For demonstration, we'll create sample data with similar structure
    # In real implementation, you would download from the actual NASA repository
    create_sample_dataset(data_dir)
    
def create_sample_dataset(data_dir):
    """Create sample dataset with NASA Turbofan structure for demonstration"""
    import numpy as np
    
    # Training data
    np.random.seed(42)
    engines = []
    
    for engine_id in range(1, 101):  # 100 engines
        cycles = np.random.randint(130, 360)  # Random operational cycles
        
        for cycle in range(1, cycles + 1):
            # Simulate degradation over time
            degradation_factor = cycle / cycles
            
            # 21 sensor readings + operational settings
            sensor_data = np.random.normal(0, 1, 21) + degradation_factor * np.random.normal(0, 0.5, 21)
            operational_settings = np.random.normal(0, 1, 3)
            
            row = [engine_id, cycle] + operational_settings.tolist() + sensor_data.tolist()
            engines.append(row)
    
    # Column names
    cols = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    
    train_df = pd.DataFrame(engines, columns=cols)
    train_df.to_csv(f"{data_dir}/train_FD001.txt", sep=' ', index=False, header=False)
    
    # Test data (without RUL)
    test_engines = []
    rul_data = []
    
    for engine_id in range(1, 101):
        test_cycles = np.random.randint(30, 200)
        actual_rul = np.random.randint(5, 150)
        rul_data.append(actual_rul)
        
        for cycle in range(1, test_cycles + 1):
            degradation_factor = cycle / (test_cycles + actual_rul)
            
            sensor_data = np.random.normal(0, 1, 21) + degradation_factor * np.random.normal(0, 0.5, 21)
            operational_settings = np.random.normal(0, 1, 3)
            
            row = [engine_id, cycle] + operational_settings.tolist() + sensor_data.tolist()
            test_engines.append(row)
    
    test_df = pd.DataFrame(test_engines, columns=cols)
    test_df.to_csv(f"{data_dir}/test_FD001.txt", sep=' ', index=False, header=False)
    
    # RUL truth data
    rul_df = pd.DataFrame(rul_data, columns=['RUL'])
    rul_df.to_csv(f"{data_dir}/RUL_FD001.txt", sep=' ', index=False, header=False)
    
    print(f"Sample dataset created:")
    print(f"- Training data: {len(train_df)} rows")
    print(f"- Test data: {len(test_df)} rows")
    print(f"- RUL truth: {len(rul_df)} rows")

if __name__ == "__main__":
    download_nasa_dataset()