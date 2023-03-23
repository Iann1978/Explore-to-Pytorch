

import torch.nn as nn

alexnet = nn.Sequential(
    nn.Conv2d(3, 96, 11,stride=4),
    nn.ReLU(),
    nn.MaxPool2d(3,stride=2),
    nn.LocalResponseNorm(5,k=2),
    
    nn.Conv2d(96, 256, 5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=2),
    nn.LocalResponseNorm(5, k=2),

    nn.Conv2d(256, 384, 3, padding=1),
    nn.ReLU(),

    nn.Conv2d(384, 384, 3, padding=1),
    nn.ReLU(),

    nn.Conv2d(384, 256, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=2),


    nn.Conv2d(256, 4096, 6),
    nn.ReLU(),
    nn.Dropout(),

    nn.Flatten(),
    nn.Linear(4096, 1000)
    # nn.Dropout(),
    #
    # nn.Linear(4096, 1000)

    # nn.MaxPool2d(3, stride=2),
)