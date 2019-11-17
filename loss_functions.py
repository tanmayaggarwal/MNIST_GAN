# define the loss functions

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

def real_loss(D_out, smooth=False):
    # compare logits to real labels
    # smooth labels if smooth = True
    if smooth==True:
        labels = torch.ones(D_out.size(0)) * 0.9
    else:
        labels = torch.ones(D_out.size(0))

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    # compare logits to fake labels
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.zeros(D_out.size(0))
    loss = criterion(D_out.squeeze(), labels)
    return loss
