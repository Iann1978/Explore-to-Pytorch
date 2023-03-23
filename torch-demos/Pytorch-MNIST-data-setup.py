# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 08:49:48 2020

@author: Iann

https://pytorch.org/tutorials/beginner/nn_tutorial.html
"""

from pathlib import Path
import requests

DATA_PATH= Path("data")
PATH = DATA_PATH/"MNIST"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH/FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (PATH/FILENAME).open("wb").write(content)
        
    
import pickle
import gzip

with gzip.open((PATH/FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

import torch
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

print(x_train,y_train)
print(x_train.shape)

print(y_train.min(), y_train.max())

import math
weights = torch.randn(784,10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)


bs = 64

xb = x_train[0:bs]
xb,xb.shape
preds = model(xb)
preds[0] , preds.shape
print(preds[0], preds.shape)


yb = y_train[0:bs]
print(yb)
print(yb.shape)
print(y_train.shape)

range(yb.shape[0])


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

print(loss_func(preds, yb))

cc = preds[0:64, (0,1,2)]
print(cc.shape)
print(cc[0])
print

dd = torch.randn((2,2))
print (dd)

print (dd.mean())


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))





from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()


print(loss_func(model(xb), yb), accuracy(model(xb), yb))
