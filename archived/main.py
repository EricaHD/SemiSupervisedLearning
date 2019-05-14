import torch
import torch.nn.functional as F
import torch.optim as optim

from load import get_train_loader, get_test_loader, get_unsup_loader
from cnn_model import CNN

######################################################
# PARSE ARGUMENTS
######################################################

######################################################
# SET UP
######################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################
# LOAD DATA
######################################################

train_loader = get_train_loader('/scratch/ehd255/ssl_data_96/supervised/train/', batch_size=32)
# unsup_loader = get_unsup_loader('/scratch/ehd255/ssl_data_96/unsupervised/', batch_size=32)
test_loader = get_test_loader('/scratch/ehd255/ssl_data_96/supervised/val/', batch_size=32)

######################################################
# BUILD MODEL
######################################################

model = CNN()
model.to(device)

######################################################
# TRAIN MODEL
######################################################

def train(num_epochs=10):

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(num_epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )

train()

######################################################
# TEST MODEL
######################################################

def test():

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
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
