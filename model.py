# define the model

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # flatten image
        x = x.view(-1, 28*28)

        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.lrelu(self.fc2(x))
        x = self.dropout(x)
        x = self.lrelu(self.fc3(x))
        x = self.dropout(x)

        x = self.output(x)
        return x


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.output = nn.Linear(hidden_dim*4, output_size)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # pass x through all layers
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.lrelu(self.fc2(x))
        x = self.dropout(x)
        x = self.lrelu(self.fc3(x))
        x = self.dropout(x)
        # final layer should have tanh applied
        x = F.tanh(self.output(x))
        return x
