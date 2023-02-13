import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

n_samples, n_feature = x.shape
print(n_samples, n_feature)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1234)

#scale
sc = StandardScaler() # for logistic regression, you want 0 mean
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshape tensor for y
y_train= y_train.view(y_train.shape[0],1)
y_test= y_test.view(y_test.shape[0],1)

#1) model
#linear combination of weights and bias and put sigmoid function at the end
class LogisticRegression(nn.Module):

    def __init__ (self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_feature)

#2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() #Binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

#3) training loop
num_epochs = 100
for epoch in range (num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    #backward_pass
    loss.backward()
    #update
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print (f'epoch :{epoch+1}, loss = {loss.item():.4f}')

#do evalation here
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc= y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy= {acc:.4f}')