import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

script_dir = os.path.dirname(os.path.abspath(__file__))


train_data_dir = os.path.join(script_dir, '../dataset/train')
val_data_dir = os.path.join(script_dir, '../dataset/validation')

# Create datasets
train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Load ResNet50 pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all layers except the last layer
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for 4 output classes (severity 1-4)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)


# Define loss and optimizer

# Cross-entropy measures how far the predicted probabilities are from the true class.
criterion = nn.CrossEntropyLoss() 

# An optimizer in machine learning is an algorithm that adjusts the weights (parameters) of a model to minimize the loss function during training
optimizer = Adam(model.fc.parameters(), lr=0.001) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        validate_model(model, val_loader)

# Validation
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)


# Save the model
torch.save(model.state_dict(), 'resnet50_mud_blisters.pth')

# To load the model later
model.load_state_dict(torch.load('resnet50_mud_blisters.pth'))


from PIL import Image

def predict_image(model, image_path):
    model.eval()
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    return predicted.item()  # Returns the predicted class (1-4 for severity)

# script_dir = os.path.dirname(os.path.abspath(__file__))


# Example of using the predict function
predicted_severity = predict_image(model, 'dataset/train/2/oyster_image_example.jpg')
print(f'Predicted Severity: {predicted_severity}')

