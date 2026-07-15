import os
import time
import glob
import numpy as np
import pandas as pd
import onnxruntime as ort

def analyze_thermal_state(raw_frame, is_person_detected):
    if not is_person_detected:
        return "EMPTY", "SYSTEM OFF", 0.0
        
    human_pixels = raw_frame[raw_frame > 28.0]
    if len(human_pixels) == 0:
        return "NOISE", "SKIP", 0.0
        
    core_temp = np.max(human_pixels)     
    avg_temp = np.mean(human_pixels)     
    
    if core_temp > 34.5 or avg_temp > 32.0:
        return "HOT", "MAX COOLING", core_temp
    elif core_temp < 32.0 or avg_temp < 29.5:
        return "COLD", "CLOSE VENTS / HEAT", core_temp
    else:
        return "STABLE", "MAINTAIN CURRENT", core_temp

# --- INDEPENDENT DATA LOADER (PURE NUMPY, NO TORCH) ---
class PureNumpyThermalLoader:
    def __init__(self, repo_path):
        self.frames = []
        self.labels = []
        csv_files = glob.glob(os.path.join(repo_path, '**', '*mlx90640*.csv'), recursive=True)
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] >= 768:
                    pixel_data = numeric_df.iloc[:, -768:].values
                    for row in pixel_data:
                        self.frames.append(row.reshape(24, 32))
                        self.labels.append(0 if 'none' in file.lower() else 1)
            except:
                pass
                
        self.frames = np.array(self.frames)
        self.baseline = np.mean(self.frames[:10], axis=0) if len(self.frames) > 0 else 0

    def get_frame(self, idx):
        raw_frame = self.frames[idx]
        frame_clean = raw_frame - self.baseline
        f_min, f_max = np.min(frame_clean), np.max(frame_clean)
        frame_norm = (frame_clean - f_min) / (f_max - f_min + 1e-8)
        
        input_data = np.expand_dims(frame_norm, axis=(0, 1)).astype(np.float32)
        return input_data, raw_frame, self.labels[idx]

# --- MULTI-ZONE ONNX SYSTEM ---
def start_multi_zone_onnx_system():
    print("[SERVER] Booting Edge HVAC Dashboard (Pure ONNX/Numpy)...")
    
    onnx_path = 'models/thermal_model.onnx'
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("ONNX model not found. Please run export_model.py first to generate the ONNX model.")
        
    ort_session = ort.InferenceSession(onnx_path)
    
    repo_path = os.path.join('data', 'real_github', 'FIR-Image-Action-Dataset')
    dataset = PureNumpyThermalLoader(repo_path)
    total_frames = len(dataset.frames)
    
    # Find the first index where a person is detected (label == 1)
    start_idx = 0
    for i, label in enumerate(dataset.labels):
        if label == 1:
            start_idx = i
            break
            
    # Define zones with their starting indices
    zones = [
        {"id": "ZONE A (Cửa vào)", "idx": start_idx},
        {"id": "ZONE B (Trung tâm)", "idx": (start_idx + 1500) % total_frames},
        {"id": "ZONE C (Góc khuất)", "idx": (start_idx + 3500) % total_frames}
    ]

    print("\n" + "="*80)
    print(f"| {'MULTI-ZONE HVAC SYSTEM - ONNX ENGINE (NO-TORCH)':^76} |")
    print("="*80)
    print(f"| {'ZONE NAME':<18} | {'GUEST STATE':<15} | {'TEMPERATURE':<10} | {'HVAC COMMAND':<25} |")
    print("="*80)

    print("\n" * 4) 

    while True:
        print("\033[F" * 5, end="") # Dashboard Real-time
        
        for zone in zones:
            current_idx = zone["idx"]
            
            # 1. Load input data and raw frame from the dataset
            input_data, raw_frame_numpy, _ = dataset.get_frame(current_idx)
            
            # 2. ONNX Inference
            outputs = ort_session.run(None, {'input_frame': input_data})
            predicted_class = np.argmax(outputs[0], axis=1)[0]
            is_person = bool(predicted_class == 1)
            
            # 3. Analyze thermal state based on the raw frame and person detection
            state, action, core_temp = analyze_thermal_state(raw_frame_numpy, is_person)
            
            # 4. Print the dashboard for each zone
            temp_str = f"{core_temp:.1f}°C" if core_temp > 0 else "---"
            print(f"| {zone['id']:<18} | {state:<15} | {temp_str:<10} | {action:<25} |")
            
            # 5. Update the index for the next frame in a circular manner
            zone["idx"] = (current_idx + 1) % total_frames
            
        print("-" * 80)
        time.sleep(0.5)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    start_multi_zone_onnx_system()