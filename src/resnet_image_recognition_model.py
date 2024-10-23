import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

# transforms, resizes, flips, rotates, and normalizes the images then converts them into tensors 
# so it can go into the model
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# same thing as above but for the validation dataset
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# finds and gets the directory path of where the python file is
script_dir = os.path.dirname(os.path.abspath(__file__))

# gets the directory path of where the validation and train dataset are
train_data_dir = os.path.join(script_dir, '../dataset/train')
val_data_dir = os.path.join(script_dir, '../dataset/validation')

# Create datasets from the image directories and transforms them
train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=val_transforms)

# Create data loaders, but shuffles the images for train so it is not looking at the 
# severity levels in order, batch_size is the number of images it takes when passing 
# through the neural network
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load ResNet50 pre-trained model
model = models.resnet50(pretrained=True)

# model = models.resnet50(pretrained=False)  # Load without pre-trained weights
# model.load_state_dict(torch.load('resnet50_mud_blisters.pth'))

# Freeze all layers except the last layer, prevents from changing the parameters of the
# original resnet model
for param in model.parameters():
    param.requires_grad = False

# gets all the different possible classes of information from previous layers and constructs 
# and reduces it to the last layer of five classses (severity 0-4)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)


# Define loss and optimizer

# Cross-entropy measures how far the predicted probabilities are from the true class. In other words,
# how wrong is it
criterion = nn.CrossEntropyLoss() 

# An optimizer in machine learning is an algorithm that adjusts the weights (parameters) of a model to minimize 
# the loss function during training, it is used for learning purposes, how fast it should learn
optimizer = Adam(model.fc.parameters(), lr=0.001) 

# see if there is a GPU, if not use the CPU and puts the model into the specified hardware device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# intializing function to train the model (this is when the model learns)
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1):
    # epochs is the number of cycles it learns
    for epoch in range(num_epochs):
        model.train()
        # variables to keep track of stats
        running_loss = 0.0
        correct = 0
        total = 0
        
        # goes through images per patch from the train dataset
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass to get model output
            outputs = model(images)

            # computation for the loss of predicted value to the actual value
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training loss and accuracy (more stats)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # more stats of how the model is doing and how it is adjusting itself after each batch learned
        epoch_loss = running_loss / len(train_loader.dataset)

        # accuracy is based off predictions of training
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # time to test the model after learning
        validate_model(model, val_loader)

# Validation function to see how well the model does when it is learning
def validate_model(model, val_loader):
    # have the model evaluate itself thorugh testing
    model.eval()
    correct = 0
    total = 0
    
    # disabling learning and adjusting since the model is being tested, no learning
    with torch.no_grad():
        # iterates through images through the validation dataset
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # get the model's answers from the images and see if it is correct
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')

# Train the model using our frain function
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# set the model to evaluation mode 
model.eval()

# Function to display image, prediction, and label
def show_images_predictions(dataloader):
    for images, labels in dataloader:
        # Have the model make the prediction
        with torch.no_grad():
            output = model(images)
        _, predicted_classes = output.max(1)

        # Display the image and its predicted and actual labels
        for i in range(images.size(0)):
            # Denormalize the image
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.imshow(img)  # Convert tensor to image

            # checking to see if the model prediction and the actual label match so the plot can be saved in the right directory
            # for model analysis
            if predicted_classes[i].item() == labels[i].item():
                plt.title(f'Predicted: {predicted_classes[i].item()}, Actual: {labels[i].item()}', color='green')
                plt.savefig(f'src/correctSeverityPrediction/oyster_{i}.jpg')
            else:
                plt.title(f'Predicted: {predicted_classes[i].item()}, Actual: {labels[i].item()}', color='red')
                plt.savefig(f'src/wrongSeverityPrediction/oyster_{i}.jpg')

            plt.show()

show_images_predictions(val_loader)


# Save the model
torch.save(model.state_dict(), 'src/testingModals/resnet50_mud_blisters.pth')
