
import torch.nn as nn

lenet5 = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5),
    nn.MaxPool2d(2),
    # nn.Conv2d(16, 120, 1),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10),
    nn.Softmax(dim=0)
    # nn.Conv2d(120, 120, 1),
)

lenet5_with_relu = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # nn.Conv2d(16, 120, 1),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
    nn.ReLU(),
    # nn.Softmax(dim=0)
    # nn.Conv2d(120, 120, 1),
)

lenet5_with_tanh = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 120, 5),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(120, 120),
    nn.ReLU(),
    nn.Linear(120, 10),
    nn.ReLU(),
    # nn.Softmax(dim=0)
    # nn.Conv2d(120, 120, 1),
)

def my(name):
    if name == "LeNet5":
        return lenet5
