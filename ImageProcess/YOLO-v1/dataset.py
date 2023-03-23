
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load the COCO dataset using CocoDetection

def collate_fn(x) :
    # print(x);

    a = list(zip(*x))
    b = torch.stack(a[0])
    c = a[1]

    return (b, c)



def get_dataset(batch_size:int):
    train_dataset = CocoDetection(
        root="C:/work/datasets/COCO/val2017",
        annFile="/work/datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json",
        transform=transform
    )
    val_dataset = CocoDetection(
        root="C:/work/datasets/COCO/val2017",
        annFile="/work/datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':
    import debuger
    train_dataset, val_dataset, train_loader, val_loader = get_dataset(2)

    sample = val_dataset[0]
    image, target = sample

    print(type(image))
    print(type(target), type(target[0]), list(target[0].keys()))

    print("Hello")
    print("Hello")
    print("Hello")
    # debuger.show(sample)
    #
    # val_loader = DataLoader(dataset=val_datasets,
    #                           batch_size=2,
    #                           shuffle=True)
    #
    sample1 = next(iter(val_loader))

    print(sample1)
