# train the model

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from loss_functions import real_loss, fake_loss

def train(z_size, D, G, train_loader, d_optimizer, g_optimizer):
    # training hyperparams
    num_epochs = 10

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 400

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    D.train()
    G.train()
    for epoch in range(num_epochs):

        for batch_i, (real_images, _) in enumerate(train_loader):

            batch_size = real_images.size(0)

            ## rescaling step ##
            real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================

            # 1. Train with real images
            d_optimizer.zero_grad()
            d_output = D.forward(real_images)

            # Compute the discriminator losses on real images
            # use smoothed labels
            d_real_loss = real_loss(d_output, smooth=True)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            d_fake = D(fake_images)
            d_fake_loss = fake_loss(d_fake)

            # add up real and fake losses and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()


            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================

            # 1. Train with fake images and flipped labels
            g_optimizer.zero_grad()

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            d_fake = D(fake_images)
            g_loss = real_loss(d_fake)

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))

        # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to train mode


    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return D, G, losses
