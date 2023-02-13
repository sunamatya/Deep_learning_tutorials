# transforms for tensors, for images, conversion, compose multiple transform
# check out the documentations

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform =None):
        #data loading
        self.transform= transform
        xy= np.loadtxt('../data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # self.x = torch.from_numpy(xy[:,1:]) #slicing (all samples and not the very first column 1 to end 1:
        # self.y = torch.from_numpy(xy[:, [0]]) #n_samples, 1 (all samples but very first column)
        self.x = xy[:,1:] #slicing (all samples and not the very first column 1 to end 1:
        self.y = xy[:, [0]] #n_samples, 1 (all samples but very first column)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample= self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform: # multiplicaiton transform
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        input, target = sample
        input*= self.factor
        return input, target

dataset = WineDataset(transform=ToTensor()) # if we write none, then it becomes a numpy array
first_data = dataset[0]
# print(first_data)
features, labels = first_data
print(features)
print(labels)

composed = torchvision.transforms.Compose({ToTensor(), MulTransform(2)}) #Note check for capitals
dataset = WineDataset(transform=composed)
first_data = dataset[0]
# print(first_data)
features, labels = first_data
print(features)
print(labels)
# #dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
# train_loader = DataLoader(dataset,
#                           batch_size=4,
#                           shuffle=True,
#                           num_workers=0)
#
#
# # datatiter = iter(dataloader)
# # data= datatiter.next()
# # features, labels = data
# # print(features)
# # print(labels)
#
# #training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples/4)
# print(total_samples)
# print(n_iterations)
#
# for epoch in range(num_epochs):
#     for i, (input, labels) in enumerate(train_loader):
#         #forward
#         if (i+1) % 5 == 0:
#             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}')
#         #backward
#
#         #update weight

