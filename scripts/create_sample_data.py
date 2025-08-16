import os
import random
import math

def create_sample_dataset():
    """Create sample dataset with NASA Turbofan structure"""
    
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    random.seed(42)
    
    # Training data
    with open(f"{data_dir}/train_FD001.txt", "w") as f:
        for engine_id in range(1, 101):  # 100 engines
            cycles = random.randint(130, 360)
            
            for cycle in range(1, cycles + 1):
                degradation_factor = cycle / cycles
                
                # Generate sensor readings with degradation
                row = [str(engine_id), str(cycle)]
                
                # Operational settings (3)
                for i in range(3):
                    val = random.gauss(0, 1)
                    row.append(f"{val:.6f}")
                
                # Sensor readings (21) with degradation
                for i in range(21):
                    base_val = random.gauss(0, 1)
                    degraded_val = base_val + degradation_factor * random.gauss(0, 0.5)
                    row.append(f"{degraded_val:.6f}")
                
                f.write(" ".join(row) + "\n")
    
    # Test data
    rul_values = []
    with open(f"{data_dir}/test_FD001.txt", "w") as f:
        for engine_id in range(1, 101):
            test_cycles = random.randint(30, 200)
            actual_rul = random.randint(5, 150)
            rul_values.append(actual_rul)
            
            for cycle in range(1, test_cycles + 1):
                degradation_factor = cycle / (test_cycles + actual_rul)
                
                row = [str(engine_id), str(cycle)]
                
                # Operational settings
                for i in range(3):
                    val = random.gauss(0, 1)
                    row.append(f"{val:.6f}")
                
                # Sensor readings with degradation
                for i in range(21):
                    base_val = random.gauss(0, 1)
                    degraded_val = base_val + degradation_factor * random.gauss(0, 0.5)
                    row.append(f"{degraded_val:.6f}")
                
                f.write(" ".join(row) + "\n")
    
    # RUL truth data
    with open(f"{data_dir}/RUL_FD001.txt", "w") as f:
        for rul in rul_values:
            f.write(f"{rul}\n")
    
    print("Sample dataset created successfully!")
    print(f"Files created in {data_dir}:")
    print("- train_FD001.txt (training data)")
    print("- test_FD001.txt (test data)")
    print("- RUL_FD001.txt (remaining useful life truth)")

if __name__ == "__main__":
    create_sample_dataset()