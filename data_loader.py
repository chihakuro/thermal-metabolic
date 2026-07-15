import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# --- PULL REAL DATA FROM GITHUB ---
def download_real_github_data(data_dir):
    repo_url = "https://github.com/noahzhy/FIR-Image-Action-Dataset.git"
    repo_path = os.path.join(data_dir, "FIR-Image-Action-Dataset")
    
    if not os.path.exists(repo_path):
        print(f"Cloning real MLX90640 data from {repo_url}...")
        os.system(f"git clone {repo_url} {repo_path}")
    else:
        print("Real dataset already exists on the machine.")
        
    return repo_path

# --- CUSTOM PYTORCH DATASET READ CSV ---
class RealThermalCSVDataset(Dataset):
    def __init__(self, repo_path):
        self.frames = []
        self.labels = []
        
        print("Scanning and decoding CSV files of MLX90640...")
        csv_files = glob.glob(os.path.join(repo_path, '**', '*mlx90640*.csv'), recursive=True)
        
        for file in csv_files:
            try:
                # Read CSV by Pandas, skip lines with errors (if any)
                df = pd.read_csv(file, on_bad_lines='skip')
                
                # Filtering algorithm: Find columns containing numerical data (thermal pixels)
                numeric_df = df.select_dtypes(include=[np.number])
                
                # MLX90640 sensor has 24x32 = 768 pixels
                if numeric_df.shape[1] >= 768:
                    # Take the last 768 columns (usually the first columns are timestamp/metadata)
                    pixel_data = numeric_df.iloc[:, -768:].values
                    
                    for row in pixel_data:
                        # 1D array to 2D array (24 rows, 32 columns)
                        self.frames.append(row.reshape(24, 32))
                        
                        # Label based on file name (contains 'none' = 0, otherwise = 1)
                        label = 0 if 'none' in file.lower() else 1
                        self.labels.append(label)
            except Exception as e:
                print(f"  -> Ignoring {os.path.basename(file)} due to structure error: {e}")
                
        self.frames = np.array(self.frames)
        self.labels = np.array(self.labels)
        
        # Preprocessing: Get the first 10 frames of the dataset as the background temperature (Baseline)
        if len(self.frames) > 0:
            self.baseline = np.mean(self.frames[:10], axis=0)
            print(f"Completed! Extracted {len(self.frames)} real thermal frames.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # 1. Clean frame (Environment noise/static removal) by subtracting the baseline
        frame_clean = frame - self.baseline
        
        # 2. Min-Max Scaling to [0, 1] range for neural network input
        f_min = np.min(frame_clean)
        f_max = np.max(frame_clean)
        frame_norm = (frame_clean - f_min) / (f_max - f_min + 1e-8)
        
        # 3. Convert to PyTorch tensor and add channel dimension (1, 24, 32)
        frame_tensor = torch.tensor(frame_norm, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frame_tensor, label_tensor

# --- TESTING ---
if __name__ == "__main__":
    data_dir = os.path.join('data', 'real_github')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Pull real dataset from GitHub (if not already present)
    repo_path = download_real_github_data(data_dir)
    
    # 2. Load the dataset using the custom PyTorch Dataset class
    dataset = RealThermalCSVDataset(repo_path)
    
    # 3. Create a DataLoader to batch the data
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for batch_frames, batch_labels in dataloader:
            print(f"Size of Batch Tensor (Images) : {batch_frames.shape}") 
            print(f"Size of Batch Tensor (Labels): {batch_labels.shape}") 
            print("*** REAL-TIME DATA HAS BEEN PROCESSED! READY FOR THE MODEL! ***")
            break
    else:
        print("No sufficient thermal pixel data found in the CSV files.")