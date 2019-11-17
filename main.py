# main application file

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

# import and visualize the data
from import_data import import_data, visualize_data

batch_size, train_data, train_loader = import_data()
images, labels = visualize_data(train_loader)

# define the model
from model import Discriminator, Generator

# define model hyperparamaters

# discriminator hyperparameters
input_size = 28*28 # size of input image to discriminator
d_output_size = 1 # size of the output from discriminator (real or fake)
d_hidden_size = 32 # size of the last hidden layer in the discriminator

# generator hyperparameters
z_size = 28*28 # size of the latent vector to give to generator
g_output_size = 28*28 # size of the generator output (generated image)
g_hidden_size = 32 # size of the first hidden layer in the generator

# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

# check
print(D)
print()
print(G)

# calculate the losses
from loss_functions import real_loss, fake_loss

# update the generator and discriminator variables using optimizers
import torch.optim as optim

lr = 0.002 # learning rate for optimizers
d_optimizer = optim.Adam(D.parameters(), lr=lr)
g_optimizer = optim.Adam(G.parameters(), lr=lr)

# training
from train import train
D, G, losses = train(z_size, D, G, train_loader, d_optimizer, g_optimizer)

# plotting the training loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label = "Discriminator")
plt.plot(losses.T[1], label = "Generator")
plt.title("Training losses")
plt.legend()

# view samples of images from the generator
from view_samples import view_samples

# load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

view_samples(-1, samples) # -1 indicates final epoch's samples (the last in the list)

# sampling from a generator
# randomly generated, new latent vectors
from new_samples_from_generator import new_samples
new_samples(z_size, G)

# saving the models
torch.save({'Discriminator_state_dict': D.state_dict(), 'Generator_state_dict': G.state_dict(), 'D_optimizer_state_dict': d_optimizer.state_dict(), 'G_optimizer_state_dict': g_optimizer.state_dict()}, "saved_model.pth")
