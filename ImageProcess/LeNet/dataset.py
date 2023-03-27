
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
data_root="e:/datasets/"

def get_dataset(name:str, batch_size:int):
    def ToDevice(input_tensor, device):
        # perform some operations on the input_tensor
        output_tensor = input_tensor.to(device)
        return output_tensor

    transform = transforms.Compose([
        # you can add other transformations in this list
        # transforms.Pad(padding=2),
        transforms.Resize(32),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: ToDevice(x, device)),
    ])
    train_dataset = MNIST(root=data_root + '/MNIST', train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_root + '/MNIST', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, test_dataset, train_loader, test_loader

if __name__ == '__main__':
    import torchvision
    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    train_dataset, test_dataset, train_loader, test_loader = get_dataset("MNIST", batch_size=8)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    # matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.close()