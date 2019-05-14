import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available(), flush=True)


def load_data(batch_size, split):
    """ Method returning a data loader for labeled data """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2835, 0.2767, 0.2950))  # RGB means, RGB stds
    ])
    data = datasets.ImageFolder(f'/scratch/ehd255/ssl_data_96/supervised/{split}', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    return data_loader

print('load data...', flush=True)
train_loader = load_data(split='train', batch_size=128)
test_loader = load_data(split='val', batch_size=128)
print('loading complete', flush=True)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1000)

    def forward(self, x):
        return self.model(x)

print('load model', flush=True)
model = torch.load('/scratch/jtb470/baselines-1/res-24.pth').to(device)
# model = ResNet18().to(device)
# model = model.load_state_dict(trained_model['state_dict']).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


def train(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f'epoch {epoch+25}', flush=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+25, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()), flush=True
                )
        if epoch % 2 == 0:
            torch.save(model, f'/scratch/jtb470/baselines-4/res-{epoch+25}.pth')


def evaluate(top_k=5):
    """ Method returning accuracy@1 and accuracy@top_k """
    print(f'\nEvaluating val set...', flush=True)
    model.eval()
    n_samples = 0.
    n_correct_top_1 = 0
    n_correct_top_k = 0

    for img, target in test_loader:
        img, target = img.to(device), target.to(device)
        batch_size = img.size(0)
        n_samples += batch_size

        # Forward
        output = model(img)

        # Top 1 accuracy
        pred_top_1 = torch.topk(output, k=1, dim=1)[1]
        n_correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).int().sum().item()

        # Top k accuracy
        pred_top_k = torch.topk(output, k=top_k, dim=1)[1]
        target_top_k = target.view(-1, 1).expand(batch_size, top_k)
        n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()

    # Accuracy
    top_1_acc = n_correct_top_1/n_samples
    top_k_acc = n_correct_top_k/n_samples

    # Log
    print(f'top 1 accuracy: {top_1_acc:.4f}', flush=True)
    print(f'top {top_k} accuracy: {top_k_acc:.4f}', flush=True)



train(25)
torch.save(model, '/scratch/jtb470/baselines-4/res-final.pth')
evaluate()


