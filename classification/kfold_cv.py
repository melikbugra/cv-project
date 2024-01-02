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


BASE = "neww/train/"
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


train_data_path = "neww/train"
val_data_path = "neww/validation"
test_data_path = "neww/test"

train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)
kfold_dataset = ConcatDataset([train_dataset, val_dataset])


batch_size = 128

# train_loader = DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
# )
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_loader = DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
# )

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
num_epochs = 20
k_folds = 5

# KFold provides train/test indices to split data in train/test sets
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print("--------------------------------")


# Define the learning rate scheduler
def get_scheduler(optimizer):
    return StepLR(optimizer, step_size=4, gamma=0.1)


# Start print
print("--------------------------------")

# Prepare to track the loss values
fold_performance = {}

# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kfold.split(kfold_dataset)):
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        kfold_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4
    )
    val_loader = DataLoader(
        kfold_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.3)
    scheduler = get_scheduler(optimizer)
    print(f"Learning Rate: {scheduler.get_last_lr()}")

    # Initialize lists to track per-epoch loss and accuracy
    train_losses, val_losses = [], []

    # Run the training loop for defined number of epochs
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        avg_val_loss = running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()}")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    # Save model after training
    save_path = f"./model-fold-{fold}.pth"
    torch.save(model.state_dict(), save_path)

    # Store the performance metrics for this fold
    fold_performance[fold] = {"train_loss": train_losses, "val_loss": val_losses}

    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy for fold {fold}: {accuracy:.2f}%")
    print("--------------------------------")

print("Training complete")

# Plotting the loss curves
for fold, performance in fold_performance.items():
    plt.figure(figsize=(12, 6))
    plt.plot(performance["train_loss"], label="Train Loss")
    plt.plot(performance["val_loss"], label="Validation Loss")
    plt.title(f"Loss Plot for Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
