import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroThermalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MicroThermalCNN, self).__init__()
        
        # Convolutional Layer 1: Receive thermal image (1 channel), output 8 channels. Kernel 3x3, Padding 1 to maintain spatial dimensions.
        # Input shape: (1, 24, 32) -> Output shape: (8, 24, 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8) # Batch normalization helps the model converge faster
        
        # Max Pooling Layer 1: Reduce spatial dimensions by half
        # Input shape: (8, 24, 32) -> Output shape: (8, 12, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2: Increase feature depth from 8 to 16 channels.
        # Input shape: (8, 12, 16) -> Output shape: (16, 12, 16)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Max Pooling Layer 2: Continue reducing spatial dimensions by half
        # Input shape: (16, 12, 16) -> Output shape: (16, 6, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer
        # Flatten data from 3D matrix (16, 6, 8) to 1D array with size: 16 * 6 * 8 = 768 elements
        self.fc1 = nn.Linear(in_features=16 * 6 * 8, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes) # Output number of classes (2 classes)
        
        self.dropout = nn.Dropout(p=0.2) # Reduce noise and prevent overfitting during training

    def forward(self, x):
        # Going through conv layer 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Going through conv layer 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flattening the data
        x = x.view(-1, 16 * 6 * 8)
        
        # Going through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# --- FUNCTION TO INSPECT MODEL SIZE AND DATA FLOW ---
def inspect_model():
    # Model initialization
    model = MicroThermalCNN(num_classes=2)
    
    # Calculate total number of parameters (weights + biases)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params:,}")
    
    # Create a mock batch with the correct shape for forward pass
    mock_batch = torch.randn(16, 1, 24, 32)
    output = model(mock_batch)
    print(f"Input Shape  : {mock_batch.shape}")
    print(f"Output Shape : {output.shape}")
    print("-----------------------------------")

if __name__ == "__main__":
    inspect_model()