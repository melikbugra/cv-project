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
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold


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
kfold_dataset = ConcatDataset([train_dataset, val_dataset])


batch_size = 32

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
num_epochs = 2
k_folds = 5

# KFold provides train/test indices to split data in train/test sets
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print("--------------------------------")

# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kfold.split(kfold_dataset)):
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        kfold_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4
    )
    val_loader = DataLoader(
        kfold_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4
    )

    # Init the neural network
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(num_ftrs, len(vehicle_types))
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-4)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f"Starting epoch {epoch+1}")

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Get inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 9:
                print("Loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 10))
                current_loss = 0.0

    # Process is complete.
    print("Training process has finished. Saving trained model.")

    # Print about testing
    print("Starting testing")

    # Saving the model
    save_path = f"./model-fold-{fold}.pth"
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for inputs, labels in val_loader:
            # Get inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # Generate outputs
            outputs = model(inputs)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print accuracy
        print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
        print("--------------------------------")

print("Training complete")
