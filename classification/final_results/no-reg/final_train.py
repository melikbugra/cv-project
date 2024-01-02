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
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR


BASE = "../../classification_data/train/"
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


train_data_path = "../../classification_data/train"
val_data_path = "../../classification_data/validation"
test_data_path = "../../classification_data/test"

train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)
train_dataset = ConcatDataset([train_dataset, val_dataset])


batch_size = 128

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(vehicle_types))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
num_epochs = 20

# Start print
print("--------------------------------")


def get_scheduler(optimizer):
    return StepLR(optimizer, step_size=4, gamma=0.1)


optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.3)
scheduler = get_scheduler(optimizer)
print(f"Initial Learning Rate: {scheduler.get_last_lr()}")

train_losses, test_losses = [], []

for epoch in range(0, num_epochs):
    print(f"Starting epoch {epoch+1}")

    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_test_loss = running_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Step the scheduler
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()}")

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
    )

# Save model after training
save_path = "./model-final-no-reg.pth"
torch.save(model.state_dict(), save_path)
print("Training complete")

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
