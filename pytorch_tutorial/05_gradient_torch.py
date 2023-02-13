#import numpy as np
# 1) Design model (input, output_size, forward_pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradient
#  - update_weights
import torch

#f= w*x

#f = 2*x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)

#model prediction
def forward(x):
    return w*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# manual gradient
#1/N*(w*x-y)**2
#dJ/dw = 1/N 2x (w*x-y)
# def gradient (x,y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5)= {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward_pass
    #dw = gradient(X,Y,y_pred)
    l.backward() #gradient of our loss with respect with w = dl/dw

    #update weights
    with torch.no_grad():
        #w -= learning_rate*dw
        w -= learning_rate*w.grad

    #zero_graidnets
    w.grad.zero_()

    if epoch %10 == 0:
        print(f'epoch {epoch+1}:w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5): .3f}')

