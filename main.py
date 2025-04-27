# 1. Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10 
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 2. Model Definition
class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        # very basic CNN example (you'll want to improve this)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # input: grayscale (1 channel)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # output: ab channels (2 channels for color)
            nn.Tanh()  # color channels usually normalized between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 3. Data Loading (simple placeholder)
def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB
    l_channel = img[:, :, 0]  # L channel (lightness)
    ab_channels = img[:, :, 1:]  # a and b channels (color)

    # Normalize
    l = l_channel / 255.0
    ab = (ab_channels - 128) / 128.0

    l = torch.tensor(l).unsqueeze(0).float()  # [1, H, W]
    ab = torch.tensor(ab).permute(2, 0, 1).float()  # [2, H, W]

    return l, ab

# 4. Training Loop (basic version)
def train(model, optimizer, criterion, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        # Example: Load 1 image
        l, ab = load_image('path_to_your_image.jpg')
        l = l.unsqueeze(0)  # add batch dimension
        ab = ab.unsqueeze(0)

        output = model(l)
        loss = criterion(output, ab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 5. Main Function
if __name__ == "__main__":
    dataset = CIFAR10(root='./data', download=True, transform=transforms.ToTensor())
    print("Hello, world!")
    #model = ColorizationCNN()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #criterion = nn.MSELoss()

    #train(model, optimizer, criterion)