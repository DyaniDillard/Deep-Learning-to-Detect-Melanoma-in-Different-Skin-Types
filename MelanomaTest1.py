import torch
import torch.nn as nn  # Neural network modules, Loss functions, etc.
import torch.optim as optim  # Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms 
import torchvision
import os
import pandas as pd
import numpy as np
import time
from torchvision import datasets, transforms, models
from PIL import Image
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  

MODEL_NAME = f"model-{int(time.time())}"
print(MODEL_NAME)

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.df.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Set device
device = "cpu"

# Hyperparameters
in_channel = 3 
num_classes = 2
learning_rate = 0.001
batch_size = 50
num_epochs = 15

# Load Data - be sure  to change with your own file path
dataset = MelanomaDataset(
    csv_file="/Users/dyanidillard/Desktop/melanomaclassifier/fair_dark_split.csv",
    root_dir="/Users/dyanidillard/Desktop/melanomaclassifier/fair_dark_dataset",
    transform=transforms.ToTensor(),
)

# Dataset loaders and splitting dataset
train_set, test_set = torch.utils.data.random_split(dataset, [960, 240])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Pre-trained model declaration, change model if you want to use a different one
model = torchvision.models.resnet50(pretrained=True)
model.to(device)

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward propagation
        predictions = model(data)
        loss = loss_function(predictions, targets)
        
        losses.append(loss.item())
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Loss at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see the performance of the model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            predictions = model(x)
            _, m_predictions = predictions.max(1) 
            num_correct += (m_predictions == y).sum()
            num_samples += m_predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking sensitivity accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking sensitivity accuracy on Validation Set")
check_accuracy(test_loader, model)