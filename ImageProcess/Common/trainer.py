import torch

class ClassifyTaskTrainer:
    def __init__(self, net, loader, loss_fn, optimizer, device):
        self.net = net
        self.netname = net.name
        self.loader = loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, epoch):
        for banch, (images, labels) in enumerate(self.loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            pred = self.net(images)
            loss = self.loss_fn(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        torch.save(self.net.state_dict(), "runs/{}_epoch{}.pt".format(self.netname, epoch))

    def valuate_one_epoch(self, epoch):
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('{}.Test Accuracy: {:.2f}%, correct:{}, total:{}'.format(epoch, accuracy, correct, total))



