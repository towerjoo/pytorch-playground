import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def wrap_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5,.5), (.5,.5,.5,.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data", train=False,
    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    return
    img = img / 2 + .5
    npimg = img.numpy()
    try:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    except:
        print "plot NA"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = wrap_cuda(Net())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)

nepoch = 10
for epoch in range(nepoch):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(wrap_cuda(inputs)), Variable(wrap_cuda(labels))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0

print('finished training')
dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# print images
imshow(torchvision.utils.make_grid(images))

outputs = net(Variable(wrap_cuda(images)))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]]
                              for j in range(4)))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(wrap_cuda(images)))
    _, predicted = torch.max(outputs.data, 1)
    labels = wrap_cuda(labels)
    total += labels.size()[0]
    correct += (predicted == labels).sum()
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
