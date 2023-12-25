from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE = "alidata/train/"
vehicle_types = os.listdir(BASE)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet's mean and std
    ]
)
batch_size = 32
test_data_path = "alidata/test"
test_dataset = ImageFolder(root=test_data_path, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)


model_path = "model-fold-4.pth"
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(vehicle_types))
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


true_labels = []
predictions = []


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())
        predictions.extend(predicted.cpu().numpy())

# Generate the classification report
report = classification_report(true_labels, predictions, target_names=vehicle_types)
print(report)
