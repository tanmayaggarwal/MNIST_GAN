# import and visualize the data
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms

# import the data

def import_data():
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # get the training datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    # prepare data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    return batch_size, train_data, train_loader

def visualize_data(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    # get one image from the batch
    img = np.squeeze(images[0])

    fig = plt.figure(figsize = (3,3))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

    return images, labels
