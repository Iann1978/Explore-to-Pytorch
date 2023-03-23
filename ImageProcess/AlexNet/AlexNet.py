
import model
from torchsummary import summary

net = model.alexnet
summary(net, (3, 227, 227), device="cpu")

import  torchvision
from torchvision import transforms

def get_dataset():
    """
    Uses torchvision.datasets.ImageNet to load dataset.
    Downloads dataset if doesn't exist already.
    Returns:
         torch.utils.data.TensorDataset: trainset, valset
    """

    trainset = torchvision.datasets.ImageNet('/work/datasets/ImageNet/train/', split='train',
                                 target_transform=None, download=True)
    valset = torchvision.datasets.ImageNet('/work/datasets/ImageNet/val/', split='val',
                               target_transform=None, download=True)

    return trainset, valset

train_dataset, test_dataset = get_dataset()