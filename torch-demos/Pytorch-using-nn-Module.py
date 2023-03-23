# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:30:07 2020

@author: Iann

https://pytorch.org/tutorials/beginner/nn_tutorial.html
"""

from matplotlib import pyplot
import numpy as np
import pickle
import gzip
from pathlib import Path
import requests
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Define loading function
def LoadMnist():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        
    #pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    print(x_train.shape)
    
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    
    return x_train, y_train, x_valid, y_valid

# Define model class
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.lin = nn.Linear(784,10)
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(16,10,kernel_size=3,stride=2,padding=1)
        
    def forward(self, xb):
        xb = xb.view(-1,1,28,28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
    

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred,yb)
            #print (loss)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))

def get_model():
    model = MyModel()
    return model, optim.SGD(model.parameters(),lr=lr)

# Set parameters for trai
epochs = 20
bs = 64
lr = 0.1  

# Loading data
x_train, y_train, x_valid, y_valid = LoadMnist()
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

train_dl,valid_dl = get_data(train_ds, valid_ds, bs)

# Generate model 
model, opt = get_model()   

# Generate loss function
loss_func = F.cross_entropy



fit(epochs, model, loss_func, opt, train_dl, valid_dl)    









