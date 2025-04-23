import torch
import numpy as np
from fontTools.misc.iterTools import batched
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        #convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        #fully connected layer
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

        #dropout
        self.dropout = nn.Dropout(0.2)

        #output layer
        self.out = nn.LogSoftmax(dim=1)

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)
        return x

# visualize batch of training data
import matplotlib.pyplot as plt
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from tensor to image
    plt.show()

if __name__ == '__main__':
    #load data
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data.sampler import SubsetRandomSampler

    # number of sub process
    num_workes = 0
    batch_size = 20
    valid_size = 0.2
    #convert data to normalized tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # download data
    train_data = datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10(root='data', train=False, download=False, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    #prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workes)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workes)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workes)

    # specify image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # get one batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy to display
    # plot images
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        ax.set_title(classes[labels[idx]])
        #imshow(images[idx])

    # view an image in more detail
    rgb_img = np.squeeze(images[3])
    channels = ['red channel', 'green channel', 'blue channel']
    fig = plt.figure(figsize = (36, 36))
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] !=0 else 0
                ax.annotate(str(val), xy = (y,x), horizontalalignment='center', verticalalignment='center', size=8, color='white' if img[x][y] < thresh else 'black')

    #define model
    model = CNNNet()
    print(model)

    if train_on_gpu:
        model.to(device)

    summary(model, input_size=images.shape[1:], batch_size=20)

    #define loss and optimizers
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #train the model
    n_epochs = 30
    epochs_no_improve = 0
    max_epochs_stop = 3

    save_file_name = 'cifar_cnn_model.pt'
    valid_loss_min = np.inf # track changes in validation loss

    def train(model, train_loader, valid_loader, n_epcohs = 20, save_file = 'cifar_cnn_model.pt'):
        # define loss and optimizers
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters())
        epochs_no_improve = 0
        max_epochs_stop = 3
        valid_loss_min = np.inf

        for epoch in range(n_epcohs):
            #keep track of train and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0
            # train the model
            model.train()
            for i, (data, target) in enumerate(train_loader):
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                #clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass
                output = model(data)
                #calculate loss
                loss = criterion(output, target)
                #backward pass
                loss.backward()
                #perform a single optimization step
                optimizer.step()
                #update training loss
                train_loss += loss.item()

                #calculate accuracy
                ps = torch.exp(output)
                topk, topclass = ps.topk(1, dim=1)
                equals = topclass = target.view(*topclass.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                train_acc += accuracy.item()

                print(f'Epoch:{epoch} \t {100*i/len(train_loader):.2f}% complete', end = '\r')

            #validate the model
            model.eval()
            for data, target in valid_loader:
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

                ps = torch.exp(output)
                topk, topclass = ps.topk(1, dim=1)
                equals = topclass = target.view(*topclass.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                valid_acc += accuracy.item()

            #calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)

            train_acc = train_acc/ len(train_loader)
            valid_acc = valid_acc/ len(valid_loader)

            #print training and validation statistics
            print(f'\n Epoch:{epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
            print(f'Training accuracy: {100 * train_acc:.2f}%\t Validation accuracy :{100 * valid_acc:.2f}%')

            # save model if validation loss is decreased
            if valid_loss <= valid_loss_min:
                print('validation loss decreased ({:.6f}) -> {:.6f}. saving model....'.format(
                    valid_loss_min, valid_loss))
                torch.save(model.state_dict(), save_file)
                epochs_no_improve = 0
                valid_loss_min = valid_loss
            else:
                epochs_no_improve += 1
                print(f'{epochs_no_improve} epochs with no improvement.')
                if epochs_no_improve >= max_epochs_stop:
                    print('Stopping early!')
                    break

    # call train function
    train(model, train_loader, valid_loader, n_epochs, save_file_name)


