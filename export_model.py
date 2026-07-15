import os
import torch
import torch.nn as nn
from models.micro_cnn import MicroThermalCNN

def quantize_and_export():
    device = torch.device("cpu") # Quantization is only supported on CPU in PyTorch
    
    # 1. Initialization and Loading of the Original Model (Float32)
    model_fp32 = MicroThermalCNN(num_classes=2)
    model_fp32.load_state_dict(torch.load('models/thermal_model.pth', map_location=device, weights_only=True))
    model_fp32.eval()
    
    fp32_size = os.path.getsize('models/thermal_model.pth') / 1024
    print(f"[1] Size of the original model (Float32): {fp32_size:.2f} KB")

    # 2. Dynamic Quantization to Int8
    # Force quantization of Linear layers (which have the most weights) to 8-bit integers
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    quantized_path = 'models/thermal_model_int8.pth'
    torch.save(model_int8.state_dict(), quantized_path)
    
    int8_size = os.path.getsize(quantized_path) / 1024
    print(f"[2] Size after Quantization (Int8): {int8_size:.2f} KB")
    print(f"    -> Compression Ratio: Reduced by {(1 - int8_size/fp32_size)*100:.1f}%!")

    # 3. EXPORT TO ONNX FORMAT (Industry Standard)
    onnx_path = 'models/thermal_model.onnx'
    
    # Create a dummy input tensor to let ONNX "measure" the data flow dimensions
    # (1 Batch, 1 Channel, 24 Rows, 32 Columns)
    dummy_input = torch.randn(1, 1, 24, 32)
    
    torch.onnx.export(
        model_fp32,                  # Model to export
        dummy_input,                 # Dummy input for shape inference
        onnx_path,                   # Output file path
        export_params=True,          # Export the model parameters
        opset_version=11,            # ONNX opset version (high compatibility)
        do_constant_folding=True,    # Optimization: Fold constant operations
        input_names=['input_frame'], # Name of the input node
        output_names=['thermal_status'], # Name of the output node
        dynamic_axes={
            'input_frame': {0: 'batch_size'}, 
            'thermal_status': {0: 'batch_size'}
        } # Allow dynamic change of batch size
    )
    
    print(f"[3] Exported the ONNX model at: {onnx_path}")
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    quantize_and_export()