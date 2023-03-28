
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

# root = 'c:/work/datasets/'

root = 'e:/datasets/'

def get_dataset(name:str, batch_size:int):
    if name == 'CIFAR10':
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Pad(padding=2),
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: ToDevice(x, device)),
        ])

        train_dataset = CIFAR10(root + 'CIFAR10', transform=transform, train=True, download=True)
        test_dataset = CIFAR10(root + 'CIFAR10', transform=transform, train=False, download=True)
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

        train_dataset = CIFAR10(root + 'STL10/', transform=transform, train=True, download=True)
        test_dataset = CIFAR10(root + 'STL10/', transform=transform, train=False, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

        return train_dataset, test_dataset, train_loader, test_loader

    elif name == 'ImageNet':
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Pad(padding=2),
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: ToDevice(x, device)),
        ])

        train_set = ImageNet(root + 'ImageNet/', split='train', transform=transform)
        val_set = ImageNet(root + 'ImageNet/', split='val', transform=transform)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=batch_size,
                                 shuffle=True)

        return train_set, val_set, train_loader, val_loader



if __name__ == '__main__':
    train_set, val_set, train_loader, val_loader = get_dataset('ImageNet', 2)

    image, label = next(iter(val_loader))

    print(image.shape)
    print(label.shape)

    print('hello')

    # images, labels = test_loader
    # from debuger import show_dataset
    #
    # show_dataset(train_dataset)