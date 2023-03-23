
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_dataset(name:str, batch_size:int):
    if name == 'CIFAR10':
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Pad(padding=2),
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: ToDevice(x, device)),
        ])

        train_dataset = CIFAR10('/work/datasets/CIFAR10', transform=transform, train=True, download=True)
        test_dataset = CIFAR10('/work/datasets/CIFAR10', transform=transform, train=False, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

        return train_dataset, test_dataset, train_loader, test_loader
    elif name == 'STL10':
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Pad(padding=2),
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: ToDevice(x, device)),
        ])

        train_dataset = CIFAR10('/work/datasets/STL10/', transform=transform, train=True, download=True)
        test_dataset = CIFAR10('/work/datasets/STL10/', transform=transform, train=False, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

        return train_dataset, test_dataset, train_loader, test_loader