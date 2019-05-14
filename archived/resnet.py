import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from load import get_train_loader, get_test_loader

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader('/scratch/ehd255/ssl_data_96/supervised/train/', batch_size=32)
test_loader = get_test_loader('/scratch/ehd255/ssl_data_96/supervised/val/', batch_size=32)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1000)

    def forward(self, x):
        return self.model(x)


model = ResNet18()
model.to(device)

def train(num_epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )
                torch.save(model, '/scratch/jtb470/semires/res-{}.pth'.format(epoch))

train()

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        accuracy)
    )

test()


torch.save(model, '/scratch/jtb470/semires/res-final.pth')
