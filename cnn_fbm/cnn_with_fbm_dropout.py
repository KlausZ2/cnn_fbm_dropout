import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from fbm_dropout_function_for_cnn import DropoutFBM
"""
How to run this CNN:
1. Make sure you have set up a proper environment for PyTorch pn your computer.
2. Make sure you have cnn_with_fbm_dropout.py, fbm_dropout_for_cnn.py, "test" folder, and "train" folder.
3. Change the path from on line 21 and 22 from 'C:/Users/Klaus Zhang/torch/cnn_fbm/train' to your own path.
4. Type "python cnn_with_fbm_dropout.py" to run the CNN.
"""
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(root='C:/Users/Klaus Zhang/torch/cnn_fbm/train', transform=transform)
test_data = datasets.ImageFolder(root='C:/Users/Klaus Zhang/torch/cnn_fbm/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self, n_agents=[50], n_samples=100, max_iters=10, t_scale=1.0, grid_sizes=[32], device=None, dtype=None):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Apply DropoutFBM to the first convolutional layer
            #DropoutFBM(0.9, n_agents[0], n_samples, max_iters, t_scale, grid_sizes, is_conv=True, device=device, dtype=dtype),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
# Define model, loss function, and optimizer
n_agents = [35]  # Number of fibers for each DropoutFBM layer
n_samples = 1000
max_iters = 40
t_scale = 1.0
grid_sizes = (16, 16)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN(n_agents, n_samples, max_iters, t_scale, grid_sizes, device=device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")