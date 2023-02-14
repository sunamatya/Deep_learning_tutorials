#MNIST # digit classification
#Dataloader, Transformations
#Multilayer Neural Networks, activation functions
#Loss and optimizers
#Training loop
#Model evaluation
#GPU support

import torch
import torch.nn as nn
import torchvision # for dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784 # 28*28
hidden_size = 100
num_classes = 10 # digits 0-9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#MNIST
training_dataset = torchvision.datasets.MNIST(root= './data', train= True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root= './data', train= False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset= training_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset= test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap = 'gray')
#plt.show()


#Multiclass problem
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 =nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2= nn.Linear(hidden_size, num_classes)
        #self.linear2= nn.Linear(hidden_size, 1) #for binary classes

    def forward(self, x):
        out= self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device) #no need for num_class for binary classification

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #reshape our images #100, 1, 28,28
        #100, 784
        images= images.reshape(-1, 28*28).to(device)
        labels= labels.to(device)

        #forward
        output= model(images)
        loss= criterion(output, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#testing loop
with torch.no_grad():
    n_correct = 0
    n_samples= 0

    for images, labels in test_loader:
        images= images.reshape(-1, 28*28).to(device)
        labels= labels.to(device)
        outputs= model(images)

        #value, index (index is the class label)
        _,predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc =  100.0*(n_correct/n_samples)
    print("accuracy in percent", acc)

