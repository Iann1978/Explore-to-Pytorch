# https://blog.csdn.net/weixin_42067873/article/details/121143902
print("VGG demo")

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary
import trainer
import debuger
import model
import torch
import dataset

batch_size = 16
device = "cuda"
epoch = 200
learning_rate = 1e-4


train_dataset, test_dataset, train_loader, test_loader = dataset.get_dataset('STL10', batch_size=batch_size)
next(iter(train_loader))

net = model.vgg16
net.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)


summary(net, (3, 224, 224), device=device)

for e in range(epoch):
    # trainer.train_one_epoch(net, train_loader, loss_fn, optimizer, device)
    trainer.test_one_epoch(net, test_loader, device)