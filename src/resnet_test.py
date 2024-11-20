import torch

# Function to test the model 
def test_model(model, test_loader): 
    correct = 0 
    total = 0 
    
    with torch.no_grad(): 
        for images, labels in test_loader: 
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            print(f'Accuracy of the model on the test images: {100 * correct / total}%') 
            
