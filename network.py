
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from torch import optim
import time
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter



writer = SummaryWriter('AlexNet')
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),                  
        transforms.Normalize([0.406], [0.225])  
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.406], [0.225])
    ]),
}

data_train = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=data_transforms['train'])
data_test = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=data_transforms['test'])
datasets = {'train': data_train, 'test': data_test}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=100,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']
               }
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}
class_names = datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(  
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()  
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                lr = next(iter(optimizer.param_groups))['lr']
                print(lr)
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Acc', epoch_acc, epoch)
            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('Test/Loss', epoch_loss, epoch)
                writer.add_scalar('Test/Acc', epoch_acc, epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_scheduler.step()  
        print()
    writer.close()
    time_elapsed = time.time() - since
    print('Trainning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model




if __name__ == '__main__':
    net = AlexNet()
    net = net.to(device)
    lr_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    net = train_model(net, criterion, optimizer, lr_scheduler, num_epochs=200)
    plt.figure()
    plt.plot(range(200), lr_list, color='r')
    plt.show()

    torch.save(net.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')




