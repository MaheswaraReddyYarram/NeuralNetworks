import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.transforms import transforms
import torch.nn.functional as F
from Cifar10Dataloader import Cifar10Dataloader

batch_size=4

def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(model, training_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    for epoch in range(1):
        for i, data in enumerate(training_data, 0):
            # get input data
            inputs, labels = data

            # zero the gradients
            optimizer.zero_grad()

            # forward, loss, backward, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

def evaluate(model, testdata_loader):
    dataiter = iter(testdata_loader)
    images, labels = next(dataiter)

    # print images
    show_image(utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


def load_data():
    # convert the images to tensor and normalized them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = Cifar10Dataloader('/Users/mahyaa/Downloads/cifar-10/train', labels_csv, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)
    return trainloader
if __name__ == '__main__':
    # Load dataset
    labels_csv = pd.read_csv('/Users/mahyaa/Downloads/cifar-10/trainLabels.csv')
    # ic(labels_csv.head())
    # since labels are stored as text, convert them to numbers
    label_mapping = {label: idx for idx, label in enumerate(labels_csv['label'].unique())}

    # now let's encode the labels
    labels_csv.rename({"label": "label_txt"}, axis=1, inplace=True)
    labels_csv['label'] = labels_csv['label_txt'].map(label_mapping)


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load images
    model = CNN()
    training_data = load_data()
    print(model)
    train(model, training_data)

    test_dataset = Cifar10Dataloader('/Users/mahyaa/Downloads/cifar-10/test', labels_csv, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2)
    evaluate(model, test_loader)

    #evaluate model

