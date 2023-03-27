
import matplotlib.pyplot as plt
import torch

def show_dataset( dataset):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(2):
        for j in range(5):
            img, label = dataset[i * 5 + j]
            label = dataset.classes[label]
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i][j].set_title(str(label), y=-0.2)
    plt.show()

def show_some_pred(net, dataset, device):
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