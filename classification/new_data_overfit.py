import matplotlib.pyplot as plt
import random
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


BASE = "alidata/train/"
vehicle_types = os.listdir(BASE)
print(vehicle_types)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet's mean and std
    ]
)


train_data_path = "alidata/train"
val_data_path = "alidata/validation"
test_data_path = "alidata/test"

train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)


batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Freeze all layers in the network
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(vehicle_types))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-4)

num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Turn off gradients for validation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    train_loss = train_loss / len(test_loader)
    val_loss = val_loss / len(test_loader)

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}"
    )

print("Training complete")
torch.save(model.state_dict(), "new_data_overfit.pth")
