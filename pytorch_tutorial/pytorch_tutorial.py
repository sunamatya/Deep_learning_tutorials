#numpy arrays to pytorch tensors (from arrays to vectors to tensors)
import torch
# empty torch, scalar value
x= torch.empty(1)
print (x)

#1D vector with three elements
x = torch.empty(3)
print (x)

#for 2D matrix
x = torch.empty(2,3)
print (x)
#for 3D matrix
x = torch.empty(2,2,3)
print (x)
# for random value, zeros, ones.. similar to the numpy, size
x = torch.ones(2,2, dtype=torch.int) #double, float16
print (x.dtype)
print (x.size())

# construct tensor from data
x = torch.tensor([1,2,3])
print(x)

# basic operation with tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x+y
print("Adding tensors", z)
z = torch.add(x, y) # torch.sub, torch.mul, torch
# in place operation is run by _ in it
y.add_(x) # mul_, sub_,
print(y)

# slicing the data
x = torch.rand(5,3)
print(x)
print(x[:,0]) #print(x[1,:])

#to get value from only one element from tensor
print(x[0,0].item())

#reshaping
x = torch.rand(4,4)
print(x)
y = x.view(16) # 16 = 4*4
print(y)
y = x.view(-1,8) # 16 = 4*4
print(y)

#convert numpy to torch tensor
import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
a.add_(1)
print(a)
print(b)
# they both point to same memory location
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)# by default float64
a +=1
print(a)
print(b) # tensor is modified too
# this happens with your tensor is on the cpu

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device= device) # to create tensor and put it in gpu
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    #z.numpy() # TODO: shows error because numpy only handles cpu tensors
    z= z.to("cpu")

x = torch.ones(5, requires_grad= True) # this says gpu that you will need gradient for this tensor later on in optimization step
# for variable you want to modily or optimize put requires_grad as true
print(x)
