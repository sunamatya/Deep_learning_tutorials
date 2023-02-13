#import numpy as np
# 1) Design model (input, output_size, forward_pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradient
#  - update_weights
import torch
import torch.nn as nn


#f= w*x

#f = 2*x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype= torch.float32)


n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)

#model prediction
# def forward(x):
#     return w*x
#model = nn.Linear(input_size, output_size) # inputsize and output size, check the shape changes

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)


#loss = MSE
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# manual gradient
#1/N*(w*x-y)**2
#dJ/dw = 1/N 2x (w*x-y)
# def gradient (x,y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()

#print(f'Prediction before training: f(5)= {forward(5):.3f}')
print(f'Prediction before training: f(5)= {model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
#optimizer = torch.optim.SGD([w], lr= learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    #y_pred = forward(X)
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward_pass
    #dw = gradient(X,Y,y_pred)
    l.backward() #gradient of our loss with respect with w = dl/dw

    #update weights
    # with torch.no_grad():
    #     #w -= learning_rate*dw
    #     w -= learning_rate*w.grad
    optimizer.step()


    #zero_graidnets
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch %10 == 0:
        #unpack them to print
        [w,b]= model.parameters()
        #print(f'epoch {epoch+1}:w = {w:.3f}, loss = {l:.8f}')
        print(f'epoch {epoch+1}:w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item(): .3f}')

