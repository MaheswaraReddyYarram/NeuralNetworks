import time

import py7zr
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
import pandas as pd
from icecream import ic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Cifar10Dataloader import Cifar10Dataloader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def extract7zfiles():
    # extract zip files
    archive_path = ['/Users/mahyaa/Downloads/cifar-10/train.7z', '/Users/mahyaa/Downloads/cifar-10/test.7z']
    extract_path = '/Users/mahyaa/Downloads/cifar-10/'
    for i in range(2):
        with py7zr.SevenZipFile(archive_path[i], mode='r') as archieve:
            # extract all files
            archieve.extractall(extract_path)
    print('Extraction done')

class EarlyStopping():

    def __init__(self, patience=5, min_change=0.001, checkpoint_path= 'best_model.pth'):
        self.patience = patience
        self.min_change = min_change
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_change:
            self.reset()
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter +=1
            print(f"Early stopping  counter: {self.counter}/ {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        self.counter = 0

def plot_train_and_val_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')

    plt.xlabel('Number of Epochs')
    plt.ylabel("Loss Value")
    plt.title('Loss vs Epochs')
    plt.legend()
    plt .show()

def calculate_metrics(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def train(model, train_loader, val_loader, loss_fn, optimizer, early_stopper, device, n_epochs=5):
    print(f"device is {device}")
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch_num in range(n_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # training step
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward, loss, backward, optimize
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        accuracy,  precision, recall, f1 = calculate_metrics(model, val_loader, device)
        train_losses.append(train_loss)

        # validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_losses.append(val_loss)
        print(
            f"Epoch {epoch_num}/{n_epochs} |  Time: {time.time() - start_time:.2f} | Train Loss = {train_loss:.4f} |",
            f"Validation Loss: {val_loss:.4f} |",
            f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
        )


        # call early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered, best model saved at {early_stopper.checkpoint_path}")
            break
    return train_losses, val_losses

class CustomeTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,  self.image_files[idx]

class CNNImageClassification(nn.Module):
    def __init__(self, dropout_probe=0.5):
        super(CNNImageClassification, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # conv layer 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # conv layer 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_probe),
            nn.Linear(128, 10)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    # Load dataset
    labels_csv = pd.read_csv('/Users/mahyaa/Downloads/cifar-10/trainLabels.csv')
    #ic(labels_csv.head())
    # since labels are stored as text, convert them to numbers
    label_mapping = {label: idx for idx, label in enumerate(labels_csv['label'].unique()) }

    # now let's encode the labels
    labels_csv.rename({"label": "label_txt"}, axis =1, inplace=True)
    labels_csv['label'] = labels_csv['label_txt'].map(label_mapping)

    # extract 7z files
   # extract7zfiles()

    train_dir = '/Users/mahyaa/Downloads/cifar-10/train'
    test_dir = '/Users/mahyaa/Downloads/cifar-10/test'

    train_imgs = os.listdir(train_dir)
    ic('Number of training images: ', len(train_imgs))
    ic('First 5 training images: ', train_imgs[:5])
    img1 = Image.open(os.path.join(train_dir, train_imgs[0]))
    ic('train img size ',  img1.size)

    # apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img1)
    #ic('img tensor size ', img_tensor.size())
    #ic('img tensor  ', img_tensor)

    # apply transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create train data set
    train_dataset = Cifar10Dataloader(train_dir, labels_csv, train_transform)


    # initialize model, loss, optimizer
    model = CNNImageClassification(dropout_probe=0.7)
    #ic(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # split train dataset into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=2)

    # create training function
    early_stopper = EarlyStopping()

    #train_losses, val_losses = train(model, train_loader, val_loader, loss_fn, optimizer, early_stopper, device, n_epochs = 10)
    #plot_train_and_val_losses(train_losses, val_losses)

    # evaluate using trained model
    test_dataset = CustomeTestDataset(test_dir, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
    model = CNNImageClassification()
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()

    #make predictions
    all_predictions = []
    img_idxs = []
    with torch.no_grad():
        for images, idxs in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            img_idxs.extend(idxs)

    #remapping predictions to classes
    idxs_to_classes = {v: k for k, v in label_mapping.items()}

    predicted_labels = [idxs_to_classes[pred] for pred in all_predictions]
    submission_df = pd.DataFrame({
        "id": [int(file.split(".")[0]) for file in img_idxs],
        "label": predicted_labels
    })

    # Sort by ID to ensure correct order
    submission_df = submission_df.sort_values(by="id")
    ic(submission_df.head())





