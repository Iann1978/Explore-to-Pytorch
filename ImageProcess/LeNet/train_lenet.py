# https://www.cnblogs.com/wangguchangqing/p/10329402.html
# https://zhuanlan.zhihu.com/p/116181964

print('This is demo for training LeNet!!! ')

import torch
import torch.nn as nn
from torchinfo import summary
import  torchvision
import model
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataset
from trainer import train_one_epoch
from trainer import test_one_epoch
from debuger import show_some_pred


device = "cuda"
batch_size = 256
epochs = 30




train_dataset, test_dataset, train_loader, test_loader = get_dataset("MNIST", batch_size)



# Debug out put
net = model.lenet5_with_relu

summary(net, (1, 1, 32, 32), device=device)
x = torch.rand(1, 1, 32, 32).to(device)
y = net(x)
net.to(device)

images, labels = next(iter(train_loader))

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
img_grid = torchvision.utils.make_grid(images)
images = images.to(device)
pred = net(images)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(net, images)
writer.close()



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, weight_decay=0.001)



test_one_epoch(net, test_loader, device)
show_some_pred(net, test_dataset, device)
for e in range(epochs):
    train_one_epoch(net, train_loader, loss_fn, optimizer, device, e+1)
    test_one_epoch(net, test_loader, device, e+1)
    # show_some_pred(net, test_dataset)



