import torch.nn as nn

vgg16 = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),
    nn.Conv2d(256, 512, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(25088, 4096),
    nn.Dropout(),
    nn.ReLU(inplace=True),

    nn.Linear(4096, 4096),
    nn.Dropout(),
    nn.ReLU(inplace=True),

    nn.Linear(4096, 10),


)

if __name__ == '__main__':
    from torchsummary import summary
    summary(vgg16, (3, 224, 224), device="cuda")