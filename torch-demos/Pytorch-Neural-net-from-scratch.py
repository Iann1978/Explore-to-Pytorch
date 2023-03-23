# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:02:42 2020

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
        
    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    print(x_train.shape)
    
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    
    return x_train, y_train, x_valid, y_valid


# Load mnist data
x_train, y_train, x_valid, y_valid = LoadMnist()
n, c = x_train.shape
# Define model 

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


import torch.nn.functional as F

loss_func = F.cross_entropy

print(loss_func(model(xb), yb), accuracy(model(xb), yb))


bs = 64
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
