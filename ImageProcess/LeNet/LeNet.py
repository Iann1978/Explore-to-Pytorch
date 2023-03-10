# https://www.cnblogs.com/wangguchangqing/p/10329402.html
# https://zhuanlan.zhihu.com/p/116181964

print('This is demo of LeNet, release by pytorch')

import torch
from torch import Tensor
import torch.nn as nn
from torchsummary import summary
import  torchvision
from torchvision import transforms
import model
import matplotlib.pyplot as plt

device = "cuda"
batch_size = 256
epochs = 30


def ToDevice(input_tensor, device):
    # perform some operations on the input_tensor
    output_tensor = input_tensor.to(device)
    return output_tensor

transform = transforms.Compose([
    # you can add other transformations in this list
    # transforms.Pad(padding=2),
    torchvision.transforms.Resize(32),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: ToDevice(x, device)),
])
train_dataset = torchvision.datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./MNIST', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)






# net = model.lenet5
# net = model.lenet5_with_relu
net = model.lenet5_with_relu
net.to(device)
summary(net, (1, 32, 32), device=device)
x = torch.rand(1, 1, 32, 32).to(device)
y = net(x)




loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, weight_decay=0.001)

def train_one_epoch(net, loader, device):


    for i, (images, labels) in enumerate(loader):
        # print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        loss = loss_fn(pred, labels)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pred.shape)

# net.to(device)

def test_one_epoch(net, loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))


def show_some_pred(net, dataset):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(2):
        for j in range(5):
            img, label = dataset[i * 5 + j]
            img = img.view(-1, *img.shape)
            lable = net(img.to(device))
            img = torch.squeeze(img.cpu()).numpy()
            _, predicted = torch.max(lable.data, 1)
            label = predicted.cpu().numpy()
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i][j].set_title(str(label[0]), y=-0.2)
    plt.show()


test_one_epoch(net, test_loader, device)
show_some_pred(net, test_dataset)
for e in range(epochs):
    train_one_epoch(net, train_loader, device)
    test_one_epoch(net, test_loader, device)
    show_some_pred(net, test_dataset)



