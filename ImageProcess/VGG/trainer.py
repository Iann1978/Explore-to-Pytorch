
import torch

def train_one_epoch(net, loader, loss_fn, optimizer, device, epoch=0):
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
        print(i)
    torch.save(net.state_dict(), "runs/vgg16_epoch{}.pt".format(epoch))


def test_one_epoch(net, loader, device, epoch=0):
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
    print('{}.Test Accuracy: {:.2f}%, correct:{}, total:{}'.format(epoch, accuracy, correct, total))