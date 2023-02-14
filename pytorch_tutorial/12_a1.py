import torch
import torch.nn as nn
import torch.nn.functional as F

#option1
class NeuralNet(nn.Module):
    def __init__(self, input_dims, hidden_size):
        super(NeuralNet, self).__int__()
        self.linear1=nn.Linear(input_dims, hidden_size)
        self.relu= nn.ReLU()
        self.linear2=nn.Linear(hidden_size, 1)
        self.sigmoid= nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.leakyrelu= nn.LeakyReLU

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

#option2
class NeuralNet(nn.Module):
    def __init__(self, input_dims, hidden_size):
        super(NeuralNet, self).__int__()
        self.linear1=nn.Linear(input_dims, hidden_size)
        self.linear2=nn.Linear(hidden_size, 1)


    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        #torch.softmax
        #torch.tanh
        #F.leaky_relu() #TODO:note leaky relu is only available in functional API so just torch may not work
        return out