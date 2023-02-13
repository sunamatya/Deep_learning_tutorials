import torch
x = torch.randn(3, requires_grad=True)
print(x)
y = x+2 #creates comptation graph (input (x&2) and output (y) node)
#now we can use forward pass, we get output y
#pytorch will create function for us and we can find gradient with backward pass
print (y) #addbackward0
z = y*y*2
# # for scalar value
# z = z.mean() #meanbackward
# z.backward() #dz/dx
# print(z) # mulbackward0
# #vector jacobian product to get the gradient
# #J * gradient vector = gradients we need
# print(x.grad)
# # for vector value
# v= torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v) #dz/dx
# print(x.grad)

#ways to remove pytorch from creating the gradient
#requires_grad(False)
#x.detach() #create new tensor that does not require gradient
#with torch.no_grad(): # do operation after that
# x.requires_grad_(False)
# print(x)
# y = x.detach()
# print(y)
# with torch.no_grad():
#     y = x+2
#     print(y)

#must empty the gradients #TODO:very important to empty gradient
# weights = torch.ones(4, requires_grad=True)
#
# for epoch in range(3):
#     model_output = (weights*3).sum()
#     model_output.backward()
#     print(weights.grad)
#     weights.grad.zero_() #don't forget the underscore in the end to keep same containter

# weights = torch.ones(4, requires_grad=True)
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()

