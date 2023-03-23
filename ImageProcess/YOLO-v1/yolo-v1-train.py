# https://blog.csdn.net/wjytbest/article/details/116116966
# https://github.com/johnwingit/YOLOV1_Pytorch
# https://github.com/johnwingit/YOLOV1_Pytorch/
# https://github.com/z-huabao/pytorch-yolov1




import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import get_vgg16_based_model
from loss import YoloLoss_v1
from torch import  optim as optim

batch_size = 16

model = get_vgg16_based_model(num_classes=90)
criterion = YoloLoss_v1()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_dataset, val_dataset, train_loader, val_loader = get_dataset(batch_size)

for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()