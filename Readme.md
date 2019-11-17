# Generative Adversarial Network

## Overview

This repository contains code that builds a generative adversarial network (GAN) trained on the MNIST dataset, and is able to generate new handwritten digits post training.

## Brief review of GANs

GANs consist of two networks, a generator G and a discriminator D, competing against each other. The generator makes "fake" data to pass to the discriminator. The discriminator also sees real training data and predicts if the data it has received is real or fake.

   * The generator is trained to fool the discriminator, it wants to output data that looks as close as possible to real, training data.
   * The discriminator is a classifier that is trained to figure out which data is real and which is fake.

Eventually, the generator learns to make data that is indistinguishable from real data to the discriminator.

Input to the generator is called the "latent sample" or "latent vector". The latent sample is a random vector that the generator uses to construct its fake images.

## Main.py overview

The following steps are followed in the main.py file:
1. Import and visualize the data
2. Define the model
   a. Discriminator network - linear classifier with at least one hidden layer (each of which have a Leaky ReLu activation function applied to their outputs) and a sigmoid output (using a BCEWithLogitsLoss function which combines a sigmoid activation function and binary cross entropy loss in one function)
   b. Generator network - similar to the discriminator network, except using a tanh activation function to the output layer
3. Set model hyperparameters
4. Build the network
5. Calculate losses
   a. Discriminator losses - total loss is the sum of the losses for real and fake images (note the goal of the Discriminator is the classify the real image as 1 and fake image as 0)
   b. Generator losses - similar to Discriminator losses but with flipped labels (given the goal of the Generator is to get the Discriminator to classify the fake image as 1)
6. Define the optimizers
7. Train the GAN
   This involves alternating between training the discriminator and the generator
   a. Discriminator training - compute the discriminator loss on real, training images --> generate fake images --> comput the discriminator loss on fake, generated images --> add up real and fake loss --> perform backpropagation + an optimization step to update the discriminator's weights
   b. Generator training - generate fake images --> compute the discriminator loss on fake images, using flipped labels --> perform backpropagation + an optimization step to update the generator's weights
8. Saving the trained generator and sampling new images
