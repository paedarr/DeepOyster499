import torch

# Function to test the model 
def test_model(model, test_loader): 
    correct = 0 
    total = 0 
    
    with torch.no_grad(): 
        for images, labels in test_loader: 
            output = model(images)                   
            _, predicted = output.max(1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            print("the label is: ", labels, "the predicted is: ", predicted)
            print(f'Accuracy of the model on the test images: {100 * correct / total}%') 
            
