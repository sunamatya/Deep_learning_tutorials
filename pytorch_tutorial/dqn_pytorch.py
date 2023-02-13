import torch as T
import torch.nn as nn # linear layers no convulation layers
import torch.nn.functional as F # for relu activation
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__() #constructer for the base class
        self.input_dims =input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        #model
        self.fc1= nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2= nn.Linear(self.fc1_dimcs, self.fc2_dims)
        self.fc3= nn.Linear(self.fc2_dimcs, self.n_actions)
        #loss and optimizer
        self.optimizier = optim.Adam(self.parameters(), lr= lr)
        self.loss = nn.MSELoss()

        #device
        self.device= T.device('cuda:0' is T.cuda.is_available() else 'cpu')

    def forward(self, state):
        x= F.relu(self.f1(state))
        x= F.relu(self.f2(x))
        y_predicted = self.f3(x) #no activation here
        return y_predicted

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_action, max_mem_size= 100000, eps_end = 0.01, eps_dec= 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min= eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_action)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0


        self.Q_eval = DeepQNetwork(self.lr, n_action=n_action, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256) #default to 256

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        #way to store the memory

#training loop
#forward pass and loss
#backward pass
#update