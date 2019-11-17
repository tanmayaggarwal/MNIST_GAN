# function to get completely new images from the generator by using the checkpoint saved after training
# this function requires passing in a new latent vector z to get new samples

import numpy as np
import torch
import matplotlib.pyplot as plt
from view_samples import view_samples

def new_samples(z_size, G):
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()

    G.eval() # eval mode
    rand_images = G(rand_z) # generated samples

    view_samples(0, [rand_images]) # 0 indicates the first set of samples in the passed in list

    return
