
import torch

def train_one_epoch(net, loader, loss_fn, optimizer, device):


    for i, (images, labels) in enumerate(loader):
        # print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        loss = loss_fn(pred, labels)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pred.shape)

def test_one_epoch(net, loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%, correct:{}, total:{}'.format(accuracy, correct, total))