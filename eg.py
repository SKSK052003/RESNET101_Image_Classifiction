#resnet-101
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import lr_scheduler

# Directory containing the original images
input_directory = "train"

# List of classes (subdirectories within the input_directory)
classes = os.listdir(input_directory)

# Set the split ratio (e.g., 80% training, 20% validation)
split_ratio = 0.8

# Data augmentation for the training dataset
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Add rotation augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the training dataset with data augmentation
train_dataset = datasets.ImageFolder(input_directory, transform=train_transform)

# Split the dataset into training and validation sets
dataset_size = len(train_dataset)
train_size = int(split_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders for training and validation
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Transfer Learning with ResNet-101
# Load the pre-trained ResNet-101 model
resnet = models.resnet101(pretrained=True)

# Freeze all layers except the final classification layer
for param in resnet.parameters():
    param.requires_grad = False

# Modify the final classification layer for your number of classes
num_features = resnet.fc.in_features
num_classes = len(os.listdir(input_directory))

# Add dropout to the final fully connected layer
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce learning rate every 10 epochs

# Early stopping parameters
best_accuracy = 0.0
patience = 5  # Number of epochs to wait for improvement

# Training loop
num_epochs = 10
early_stopping_counter = 0

for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Validation loop
    resnet.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy for Epoch {epoch + 1}: {validation_accuracy}%")

    # Check for early stopping
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping: No improvement in validation accuracy.")
        break

    # Adjust learning rate
    scheduler.step()

print("Training and validation complete.")

# Directory containing the test images
test_directory = "New folder"

# Define the transform for test data (consistency with training transform)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label as the second element

# Create the test dataset using the custom dataset loader
test_dataset = CustomImageFolder(test_directory, transform=test_transform)

# Create a data loader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

resnet.eval()

# Lists to store predictions and true labels
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted', zero_division=1)  # Set zero_division to avoid recall warnings
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
PATH='rest.pt'
torch.save(resnet, PATH)