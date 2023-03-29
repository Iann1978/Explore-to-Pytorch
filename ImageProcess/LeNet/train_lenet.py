# https://www.cnblogs.com/wangguchangqing/p/10329402.html
# https://zhuanlan.zhihu.com/p/116181964

print('This is demo for training LeNet!!! ')
import sys

# appending the parent directory path
sys.path.append('../Common')

print(sys.path)


import torch
import torch.nn as nn
from torchinfo import summary
import  torchvision
import model
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataset
from debuger import show_some_pred
from trainer import ClassifyTaskTrainer


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

net.name = "lenet5"
trainer = ClassifyTaskTrainer(net, test_loader, loss_fn, optimizer, device)
trainer.valuate_one_epoch(0)
for e in range(epochs):
    trainer.train_one_epoch(e+1)
    trainer.valuate_one_epoch(e+1)



